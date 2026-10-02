"""Tier config: which local model fills each role, and which tier runs each stage.

The file is local to the machine and never committed. It names models, and the choice of models
depends on measurements (see sizing.py). The validator refuses anything that would send a run
somewhere unintended: a remote endpoint, an unedited example, or a stage nobody owns.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from urllib.parse import urlparse

STAGES = range(9)                 # stages 0..8 in the spec
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}  # IPv6 loopback is ::1 only
PLACEMENTS = {"chips", "cpu"}
REQUIRED_TIER_KEYS = ("role", "endpoint", "model", "placement")
SENTINEL = "CHANGE-ME"            # example config values; a run must not start on them
NO_MODEL_STAGE = 7                # image build: the supervisor waits and no model is loaded
STAGES_REQUIRING_DIAGNOSE = {2, 3, 4}  # small runs, large diagnoses
STAGE_FORBIDDING_DIAGNOSE = {7}   # no model is loaded


class TierConfigError(ValueError):
    """Raised when tier config is invalid: missing file, bad TOML, wrong values, or unsafe setup."""
    pass


@dataclass
class TierConfig:
    """Loaded tier config: tiers map to roles and models, stages map to tier assignments."""
    tiers: dict[str, dict]
    stages: dict[int, dict]


def _contains_sentinel(value):
    """Check if a string contains CHANGE-ME (case-insensitive, as substring). Callers pass str only."""
    return SENTINEL.lower() in value.lower()


def load(path) -> TierConfig:
    """Load and validate tier config from a TOML file.

    Raises TierConfigError if the file cannot be read, parsed, or contains invalid values.
    All checks validate that runs stay local: endpoints must be on this machine, no sentinel
    values remain in the config, and stages 2/3/4 have a diagnose tier for safe fallback.
    """
    # Read and parse the file, wrapping all errors in TierConfigError.
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except (OSError, UnicodeDecodeError) as e:
        raise TierConfigError(f"cannot read {path!r}: {e}") from e
    except tomllib.TOMLDecodeError as e:
        raise TierConfigError(f"cannot parse {path!r}: {e}") from e

    # Extract tiers and stages, validating their types.
    tiers_raw = raw.get("tiers", {})
    if not isinstance(tiers_raw, dict):
        raise TierConfigError(f"tiers must be a table, not {type(tiers_raw).__name__}")
    tiers = tiers_raw

    stages_raw = raw.get("stages", {})
    if not isinstance(stages_raw, dict):
        raise TierConfigError(f"stages must be a table, not {type(stages_raw).__name__}")

    # Validate stage keys: must be exactly decimal digits (0-8), no leading zeros unless "0".
    stages = {}
    for k, v in stages_raw.items():
        # Check that the key matches the decimal format: "0" through "8" only, no "+1", " 1", "٨", etc.
        if not re.fullmatch(r"[0-9]+", k) or (len(k) > 1 and k[0] == '0'):
            raise TierConfigError(f"stage key {k!r} must be a decimal integer (0-8), no leading zeros")
        n = int(k)
        # Check that stage number is in the valid range.
        if n not in STAGES:
            raise TierConfigError(f"stage {n} is outside the valid range 0-8")
        stages[n] = v

    # Validate tier definitions.
    if "none" in tiers:
        raise TierConfigError("a tier cannot be named 'none' (reserved for stage 7)")

    for name, tier in tiers.items():
        if not isinstance(tier, dict):
            raise TierConfigError(f"tier {name!r} must be a table, not {type(tier).__name__}")

        # Check required keys.
        for key in REQUIRED_TIER_KEYS:
            if key not in tier:
                raise TierConfigError(f"tier {name!r} is missing {key!r}")

        # Validate role: non-empty string, no sentinel.
        role = tier["role"]
        if not isinstance(role, str):
            raise TierConfigError(f"tier {name!r} role must be a string, not {type(role).__name__}")
        if not role.strip():
            raise TierConfigError(f"tier {name!r} role must be a non-empty string")
        if _contains_sentinel(role):
            raise TierConfigError(f"tier {name!r} still has {SENTINEL} in role")

        # Validate model: non-empty string, no sentinel.
        model = tier["model"]
        if not isinstance(model, str):
            raise TierConfigError(f"tier {name!r} model must be a string, not {type(model).__name__}")
        if not model.strip():
            raise TierConfigError(f"tier {name!r} model must be a non-empty string")
        if _contains_sentinel(model):
            raise TierConfigError(f"tier {name!r} still has {SENTINEL} in model")

        # Validate endpoint: non-empty string, local host, http/https scheme.
        endpoint = tier["endpoint"]
        if not isinstance(endpoint, str):
            raise TierConfigError(f"tier {name!r} endpoint must be a string, not {type(endpoint).__name__}")
        if not endpoint.strip():
            raise TierConfigError(f"tier {name!r} endpoint must be a non-empty string")
        if _contains_sentinel(endpoint):
            raise TierConfigError(f"tier {name!r} still has {SENTINEL} in endpoint")

        # Parse URL; urlparse can raise ValueError for invalid IPv6 (e.g., "http://[::1/v1")
        try:
            parsed = urlparse(endpoint)
            host = parsed.hostname  # This can also raise ValueError for malformed IPv6
        except ValueError as e:
            raise TierConfigError(f"tier {name!r}: endpoint {endpoint!r} is not a valid URL: {e}") from e

        # Check scheme and host (these are logic checks, not parsing errors)
        if not parsed.scheme:
            raise TierConfigError(f"tier {name!r}: endpoint must have a scheme (http or https)")
        if parsed.scheme not in ("http", "https"):
            raise TierConfigError(f"tier {name!r}: endpoint scheme must be http or https, not {parsed.scheme!r}")
        if host not in LOCAL_HOSTS:
            raise TierConfigError(
                f"tier {name!r}: endpoint host {host!r} is not on this machine; runs use local models only")

        # Validate placement: string and in allowed set.
        placement = tier["placement"]
        if not isinstance(placement, str):
            raise TierConfigError(f"tier {name!r} placement must be a string, not {type(placement).__name__}")
        if placement not in PLACEMENTS:
            raise TierConfigError(f"tier {name!r}: placement must be one of {sorted(PLACEMENTS)}, not {placement!r}")

        # Validate context_tokens if present: integer (not bool) and positive.
        if "context_tokens" in tier:
            ct = tier["context_tokens"]
            if isinstance(ct, bool) or not isinstance(ct, int):
                raise TierConfigError(f"tier {name!r}: context_tokens must be an integer (not bool), not {type(ct).__name__}")
            if ct <= 0:
                raise TierConfigError(f"tier {name!r}: context_tokens must be positive, not {ct}")

    # Check that a CPU tier exists (fallback during parking).
    if not any(t.get("placement") == "cpu" for t in tiers.values()):
        raise TierConfigError("a tier with placement 'cpu' is required (the stand-in during parking)")

    # Validate stage definitions.
    for n in STAGES:
        if n not in stages:
            raise TierConfigError(f"stage {n} has no entry")
        stage = stages[n]
        if not isinstance(stage, dict):
            raise TierConfigError(f"stage {n} must be a table, not {type(stage).__name__}")

        # Validate run: string and either "none" (stage 7 only) or a defined tier.
        run = stage.get("run")
        if run is None:
            raise TierConfigError(f"stage {n} is missing run")
        if not isinstance(run, str):
            raise TierConfigError(f"stage {n} run must be a string, not {type(run).__name__}")

        if n == NO_MODEL_STAGE:
            # Stage 7: must be exactly "none" (no model runs).
            if run != "none":
                raise TierConfigError(f"stage {n} must have run = 'none' (image build), not {run!r}")
        else:
            # Other stages: run must be a defined tier. "none" fails here too, because no tier
            # can be named "none", so it is simply not a defined tier.
            if run not in tiers:
                raise TierConfigError(f"stage {n}: run tier {run!r} is not defined")

        # Validate diagnose if present: string and a defined tier.
        if "diagnose" in stage:
            diagnose = stage["diagnose"]
            if not isinstance(diagnose, str):
                raise TierConfigError(f"stage {n} diagnose must be a string, not {type(diagnose).__name__}")
            if diagnose not in tiers:
                raise TierConfigError(f"stage {n}: diagnose tier {diagnose!r} is not defined")

        # Enforce stage-specific diagnose rules.
        if n in STAGES_REQUIRING_DIAGNOSE:
            # Stages 2, 3, 4: must have diagnose (small runs, large diagnoses).
            if "diagnose" not in stage:
                raise TierConfigError(f"stage {n} must have a diagnose tier (small runs, large diagnoses)")
        elif n in STAGE_FORBIDDING_DIAGNOSE:
            # Stage 7: no model is loaded, so no diagnose.
            if "diagnose" in stage:
                raise TierConfigError(f"stage {n} must not have a diagnose tier (no model is loaded)")

    return TierConfig(tiers=tiers, stages=stages)
