"""The defaults for `tt-orchard bringup`: config/bringup.toml, the run directory name for a model, and the
`orchard.supervisor run` argument list those two imply.

Why a file of its own: the tier loader refuses tables it does not know, so these settings cannot live in
tiers.toml. The loader here is strict in the same way. An unknown key is refused with a suggestion, a
missing required key is named, a value that still says CHANGE-ME is refused, and a wrong type is refused.
Relative paths are relative to the config file's directory, so the command gives the same answer from any
working directory.

Everything here is a pure function of the file and the model id. Nothing reads the network, gozer or the
hardware (that is orchard/preflight.py), and nothing starts a run (orchard/cli.py).
"""
from __future__ import annotations

import argparse
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from orchard.agent import SECRET_NAME
from orchard.tiers import SENTINEL, _unknown_key_message

TOP_KEYS = {"runs_root", "tiers", "cache_root", "hf_home", "operator_home", "gozer", "required_chips",
            "skills_dirs", "package_format", "package_namespace", "package_models_root", "min_free_gb",
            "env", "coder"}
CODER_KEYS = {"target", "kind", "profile", "port", "chips", "image_id"}
CODER_KINDS = ("container", "bundle")
PACKAGE_FORMATS = ("v6", "v5.1")     # the values `supervisor run --package-format` accepts

# org/name as Hugging Face writes it. Both parts start with a letter or digit, so a leading dash, a dot
# segment and a hidden directory are out; fullmatch, because `$` would let a trailing newline through.
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")


class BringupConfigError(ValueError):
    """The bringup config or the model id is not usable."""


@dataclass
class Coder:
    target: str
    port: int
    chips: int
    kind: str = "container"
    profile: str = "default"
    image_id: str | None = None


@dataclass
class BringupConfig:
    runs_root: Path
    coder: Coder
    tiers: Path
    gozer: str = "gozer"
    cache_root: Path | None = None
    hf_home: Path | None = None
    operator_home: Path | None = None
    required_chips: str | None = None
    skills_dirs: list[Path] = field(default_factory=list)
    package_format: str | None = None
    package_namespace: str | None = None
    package_models_root: Path | None = None
    min_free_gb: float | None = None
    env: dict[str, str] = field(default_factory=dict)


def _strings(value, out: list[str]) -> list[str]:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            _strings(v, out)
    elif isinstance(value, list):
        for v in value:
            _strings(v, out)
    return out


def _check_keys(where: str, table: dict, allowed: set[str]) -> None:
    for key in table:
        if key not in allowed:
            raise BringupConfigError(_unknown_key_message(where, key, allowed))


def _required(table: dict, key: str, where: str = ""):
    if key not in table:
        raise BringupConfigError(f"{where}{key} is required but missing")
    return table[key]


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BringupConfigError(f"{name} must be a non-empty string")
    return value


def _count(value, name: str, high: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or (high and value > high):
        raise BringupConfigError(f"{name} must be a whole number from 1" + (f" to {high}" if high else ""))
    return value


def load(path) -> BringupConfig:
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BringupConfigError(f"cannot read {str(path)!r}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise BringupConfigError(f"cannot parse {str(path)!r}: {exc}") from exc
    for s in _strings(raw, []):
        if SENTINEL.lower() in s.lower():
            raise BringupConfigError(f"{path.name} still has {SENTINEL} in {s!r}; edit it first")
    _check_keys("the config", raw, TOP_KEYS)
    base = path.resolve().parent

    def where(value, name: str) -> Path:
        p = Path(_text(value, name))
        return p if p.is_absolute() else base / p

    coder_raw = _required(raw, "coder")
    if not isinstance(coder_raw, dict):
        raise BringupConfigError("coder must be a table")
    _check_keys("[coder]", coder_raw, CODER_KEYS)
    kind = coder_raw.get("kind", "container")
    if kind not in CODER_KINDS:
        raise BringupConfigError(f"coder.kind must be one of {CODER_KINDS}, got {kind!r}")
    coder = Coder(
        target=_text(_required(coder_raw, "target", "coder."), "coder.target"),
        port=_count(_required(coder_raw, "port", "coder."), "coder.port", 65535),
        chips=_count(_required(coder_raw, "chips", "coder."), "coder.chips"),
        kind=kind,
        profile=_text(coder_raw.get("profile", "default"), "coder.profile"),
        image_id=_text(coder_raw["image_id"], "coder.image_id") if "image_id" in coder_raw else None,
    )

    required_chips = raw.get("required_chips")
    if required_chips is not None:
        from orchard.supervisor import chip_counts          # the same parser the run flag uses
        try:
            chip_counts(_text(required_chips, "required_chips"))
        except (argparse.ArgumentTypeError, ValueError) as exc:
            raise BringupConfigError(f"required_chips {required_chips!r} is not a list of chip counts "
                                     f"such as '2,4': {exc}") from exc

    fmt = raw.get("package_format")
    if fmt is not None and fmt not in PACKAGE_FORMATS:
        raise BringupConfigError(f"package_format must be one of {PACKAGE_FORMATS}, got {fmt!r}")
    if fmt and not raw.get("package_namespace"):
        raise BringupConfigError("package_format needs package_namespace (the namespace goes in the card and "
                                 "the publish commands)")

    env = raw.get("env", {})
    if not isinstance(env, dict) or not all(isinstance(v, str) for v in env.values()):
        raise BringupConfigError("[env] must map names to strings")
    for name in env:
        if SECRET_NAME.search(name):
            raise BringupConfigError(f"[env] {name} looks like a credential; agent shells must not get one")

    skills = raw.get("skills_dirs", [])
    if not isinstance(skills, list):
        raise BringupConfigError("skills_dirs must be a list of paths")
    min_free = raw.get("min_free_gb")
    if min_free is not None and (isinstance(min_free, bool) or not isinstance(min_free, (int, float))
                                 or min_free <= 0):
        raise BringupConfigError("min_free_gb must be a positive number")

    def opt(key):
        return where(raw[key], key) if key in raw else None

    return BringupConfig(
        runs_root=where(_required(raw, "runs_root"), "runs_root"),
        coder=coder,
        tiers=where(raw.get("tiers", "tiers.toml"), "tiers"),
        gozer=_text(raw.get("gozer", "gozer"), "gozer"),
        cache_root=opt("cache_root"), hf_home=opt("hf_home"), operator_home=opt("operator_home"),
        required_chips=required_chips,
        skills_dirs=[where(s, "skills_dirs") for s in skills],
        package_format=fmt,
        package_namespace=_text(raw["package_namespace"], "package_namespace") if "package_namespace" in raw else None,
        package_models_root=opt("package_models_root"),
        min_free_gb=float(min_free) if min_free is not None else None,
        env=dict(env),
    )


def slug(model_id: str) -> str:
    """`Cloudflare/clef` -> `cloudflare--clef`. Refuses anything that is not `org/name`, so a model id can
    never name a directory outside the runs root."""
    if not isinstance(model_id, str) or not _MODEL_ID.fullmatch(model_id):
        raise BringupConfigError(f"{model_id!r} is not a Hugging Face model id of the form org/name")
    return model_id.lower().replace("/", "--")


def run_dir(cfg: BringupConfig, model_id: str) -> Path:
    return cfg.runs_root / slug(model_id)


def supervisor_argv(cfg: BringupConfig, model_id: str, run_dir: Path | None = None, *,
                    inputs: dict[str, str] | None = None, accept_credentials: bool = False) -> list[str]:
    """The arguments after `python3 -m orchard.supervisor`. Optional flags appear only when the config
    sets them, so the supervisor's own defaults stay the defaults. `--accept-credentials-visible` is added
    only when the caller says the operator asked for it on the command line; the config cannot add it."""
    c = cfg.coder
    argv = ["run", "--model", model_id, "--run-dir", str(run_dir or globals()["run_dir"](cfg, model_id)),
            "--tiers", str(cfg.tiers), "--coder-target", c.target, "--coder-kind", c.kind,
            "--coder-profile", c.profile, "--coder-port", str(c.port), "--coder-chips", str(c.chips)]
    for flag, value in (("--coder-image-id", c.image_id), ("--required-chips", cfg.required_chips),
                        ("--cache-root", cfg.cache_root), ("--hf-home", cfg.hf_home),
                        ("--operator-home", cfg.operator_home), ("--package-format", cfg.package_format),
                        ("--package-namespace", cfg.package_namespace),
                        ("--package-models-root", cfg.package_models_root)):
        if value is not None:
            argv += [flag, str(value)]
    argv += ["--gozer", cfg.gozer]
    for d in cfg.skills_dirs:
        argv += ["--skills-dir", str(d)]
    for name, value in cfg.env.items():
        argv += ["--env", f"{name}={value}"]
    for name, value in (inputs or {}).items():
        argv += ["--input", f"{name}={value}"]
    argv.append("--unattended")        # a bring-up started here never waits for a person
    if accept_credentials:
        argv.append("--accept-credentials-visible")
    return argv
