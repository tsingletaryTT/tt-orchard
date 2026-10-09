# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The model choices the Settings view offers: which chip layout and which CPU stand-in.

A layout is one of the QuietBox 2 presets that `tt-orchard setup` writes (config/bringup.qb2-<name>.toml and
config/tiers.qb2-<name>.toml): the coder package a run boots on the chips, its port and chip count, and the
model both chip tiers name. The supervisor boots one server on the chips, so the grafter (the small tier,
which runs the stage steps) and the head grower (the large tier, which plans and diagnoses) use the same
model; a different model per chip role needs a second server, which the supervisor does not start. The
seasonal hand (the CPU tier) is any model ollama has.

`apply` edits only the `[coder]` table of bringup.toml and the `model` and `endpoint` keys of the three tiers
in tiers.toml: the stage map, the escalation table and every comment stay as they are. It checks the new
files with the supervisor's own loaders before it writes either, keeps a timestamped copy of each, and
replaces both with `os.replace`. A change applies to the next run or retry, never to a running one.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
import tomllib
from pathlib import Path

from orchard import bringup_config, tiers
from orchard.setup_machine import CODERS

CODER_FIELDS = ("target", "kind", "profile", "port", "chips")
ROLES = (("grafter", "small", "runs each stage's steps"),
         ("head grower", "large", "plans and diagnoses"),
         ("seasonal hand", "cpu", "stands in on the CPU when the chips are busy"))


class SettingsError(Exception):
    """A choice that cannot be saved; the message says why."""


def _table(text: str, name: str) -> dict:
    """One table of a TOML file whose other lines may not parse (setup's templates hold placeholders)."""
    m = re.search(rf"^\[{re.escape(name)}\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    if not m:
        return {}
    return tomllib.loads(m.group(1))


def presets(config_dir) -> list[dict]:
    config_dir = Path(config_dir)
    out = []
    for name, info in CODERS.items():
        b, t = config_dir / f"bringup.qb2-{name}.toml", config_dir / f"tiers.qb2-{name}.toml"
        if not (b.is_file() and t.is_file()):
            continue
        coder = _table(b.read_text(encoding="utf-8"), "coder")
        tier = tomllib.loads(t.read_text(encoding="utf-8"))["tiers"]
        out.append({"name": name, "note": info["note"], "weights_gb": info.get("weights_gb"),
                    "coder": {k: coder.get(k) for k in CODER_FIELDS},
                    "chip_model": tier["large"]["model"],
                    "tiers": {k: {"model": tier[k]["model"], "endpoint": tier[k]["endpoint"]} for k in ("large", "small")}})
    return out


def _read(cfg_path) -> tuple[object, dict]:
    cfg = bringup_config.load(cfg_path)
    with open(cfg.tiers, "rb") as fh:
        return cfg, tomllib.load(fh)


def current(cfg_path, config_dir, *, cpu_models, is_installed) -> dict:
    cfg, raw = _read(cfg_path)
    tier = raw.get("tiers", {})
    found = presets(config_dir)
    layout = next((p["name"] for p in found if p["coder"]["target"] == cfg.coder.target), None)
    return {
        "config": str(cfg_path), "tiers_path": str(cfg.tiers),
        "layout": layout,
        "mode": getattr(cfg, "mode", "local"),
        "lab": getattr(getattr(cfg, "lab", None), "host", None),
        "coder": {k: getattr(cfg.coder, k, None) for k in CODER_FIELDS},
        "roles": [{"role": role, "tier": t, "does": does, "model": tier.get(t, {}).get("model"),
                   "endpoint": tier.get(t, {}).get("endpoint")} for role, t, does in ROLES],
        "layouts": [{**p, "installed": bool(is_installed(p["coder"]["target"]))} for p in found],
        "cpu_models": list(cpu_models()),
    }


def _toml_value(v) -> str:
    if isinstance(v, bool) or not isinstance(v, (str, int)):
        raise SettingsError(f"cannot write {v!r} as a setting")
    return json.dumps(v) if isinstance(v, str) else str(v)


def set_values(text: str, table: str, values: dict) -> str:
    """`text` with `key = value` set for each of `values` inside `[table]`; a missing key is added after the
    table's last line. Every other line is kept as it is."""
    lines = text.splitlines(keepends=True)
    head = next((i for i, l in enumerate(lines) if re.match(rf"^\s*\[{re.escape(table)}\]\s*(#.*)?$", l)), None)
    if head is None:
        raise SettingsError(f"the file has no [{table}] table")
    end = next((i for i in range(head + 1, len(lines)) if re.match(r"^\s*\[", lines[i])), len(lines))
    todo = dict(values)
    for i in range(head + 1, end):
        m = re.match(r"^\s*([A-Za-z0-9_-]+)\s*=", lines[i])
        if m and m.group(1) in todo:
            lines[i] = f"{m.group(1)} = {_toml_value(todo.pop(m.group(1)))}\n"
    last = end
    while last > head + 1 and not lines[last - 1].strip():
        last -= 1
    lines[last:last] = [f"{k} = {_toml_value(v)}\n" for k, v in todo.items()]
    return "".join(lines)


def set_top_value(text: str, key: str, value) -> str:
    """`text` with the top-level `key = value` set: the line before the first table is replaced, or added
    as the first line."""
    lines = text.splitlines(keepends=True)
    first_table = next((i for i, l in enumerate(lines) if re.match(r"^\s*\[", l)), len(lines))
    for i in range(first_table):
        if re.match(rf"^\s*{re.escape(key)}\s*=", lines[i]):
            lines[i] = f"{key} = {_toml_value(value)}\n"
            return "".join(lines)
    return f"{key} = {_toml_value(value)}\n" + text


def apply(cfg_path, config_dir, *, layout: str, cpu_model: str, cpu_models, is_installed, clock=time.time,
          mode: str | None = None) -> dict:
    cfg_path = Path(cfg_path)
    preset = next((p for p in presets(config_dir) if p["name"] == layout), None)
    if preset is None:
        raise SettingsError(f"there is no layout named {layout!r}")
    if not is_installed(preset["coder"]["target"]):
        raise SettingsError(f"{preset['coder']['target']} is not installed here; install it with "
                            f"`tt-model pull {preset['coder']['target']}` first")
    if cpu_model not in cpu_models():
        raise SettingsError(f"{cpu_model!r} is not in ollama's list on this machine; pull it with "
                            f"`ollama pull {cpu_model}` first")
    cfg, _ = _read(cfg_path)
    tiers_path = Path(cfg.tiers)
    b_text = set_values(cfg_path.read_text(encoding="utf-8"), "coder",
                        {k: v for k, v in preset["coder"].items() if v is not None})
    t_text = tiers_path.read_text(encoding="utf-8")
    for name, v in preset["tiers"].items():
        t_text = set_values(t_text, f"tiers.{name}", v)
    t_text = set_values(t_text, "tiers.cpu", {"model": cpu_model})
    if mode is not None:
        b_text = set_top_value(b_text, "mode", mode)

    staged = []
    try:
        for path, text in ((tiers_path, t_text), (cfg_path, b_text)):
            fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".settings", dir=path.parent)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            staged.append((path, Path(tmp)))
        tmp_tiers = staged[0][1]
        tiers.load(tmp_tiers)
        # bringup.toml names its tiers file; check the new bringup.toml against the new tiers file.
        check = staged[1][1].with_suffix(".check")
        line = f"tiers = {json.dumps(str(tmp_tiers))}"
        check.write_text(re.sub(r"(?m)^tiers\s*=.*$", line, b_text) if re.search(r"(?m)^tiers\s*=", b_text)
                         else line + "\n" + b_text, encoding="utf-8")
        try:
            bringup_config.load(check)
        finally:
            check.unlink(missing_ok=True)
    except (tiers.TierConfigError, bringup_config.BringupConfigError, OSError, ValueError) as exc:
        for _, tmp in staged:
            tmp.unlink(missing_ok=True)
        raise SettingsError(f"the new settings were not saved: {exc}") from exc

    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(clock()))
    backups = []
    for path, tmp in staged:
        bak = path.with_name(f"{path.name}.bak-{stamp}")
        shutil.copy2(path, bak)
        backups.append(str(bak))
        os.replace(tmp, path)
    return {"layout": layout, "cpu_model": cpu_model, "mode": mode, "backups": backups}
