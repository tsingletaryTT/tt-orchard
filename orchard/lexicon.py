"""The orchard vocabulary: what each state, stage and actor is called when a person reads the output.

The names are flavour. The real state and stage names always stay on the page next to them, because the
docs, the operator runbook and the ledger use the real ones. A bring-up is a graft: the nearest supported
model is the rootstock, the new model's weights are the scion, and a model that needs new code is planted
from seed. Tests tie this table to `status.STATES` and `stages.STAGES`, so a new state or stage cannot
appear without an entry.
"""
from __future__ import annotations

from typing import NamedTuple

from orchard.ui import Style


class StateEntry(NamedTuple):
    emoji: str
    role: str       # a key of ui.ROLES
    phrase: str


class StageEntry(NamedTuple):
    emoji: str
    name: str


# One entry for each of status.STATES.
STATES = {
    "running": StateEntry("🌤", "good", "growing"),
    "paused": StateEntry("🌧", "warn", "rain delay"),
    "ready-for-operator-review": StateEntry("🍎", "good", "ripe"),
    "aborted": StateEntry("🍂", "dim", "fallen"),
    "stopped-or-crashed": StateEntry("⛈", "alarm", "storm damage"),
    "not-started": StateEntry("🌰", "dim", "seed"),
}

# One entry for each stage number in stages.STAGES.
STAGES = {
    0: StageEntry("🔭", "survey"),
    1: StageEntry("🧪", "soil test"),
    2: StageEntry("🌱", "graft"),
    3: StageEntry("🌳", "trunk"),
    4: StageEntry("🌿", "rows"),
    5: StageEntry("🚪", "farm gate"),
    6: StageEntry("🍯", "taste test"),
    7: StageEntry("📦", "crate"),
    8: StageEntry("🧺", "harvest"),
}

# Who does what. The README's "who is who" table is written from this one.
ACTORS = {
    "orchardist": "the supervisor. It tends every row and answers for the whole run.",
    "grafter": "the coder model. It does the hands-on work in each stage.",
    "head grower": "the large tier. It is called in for plans and hard diagnoses.",
    "seasonal hand": "the CPU tier. It fills in while the chips are busy.",
    "sheepdog": "the watchdog. It watches the rows and acts only on what it launched.",
    "almanac": "the ledger. It is append-only, and every step and decision is written in it.",
    "gate": "the gozer lease. One party at a time goes through.",
    "shed": "park and restore. The coder is put away so the chips can be used, then brought back.",
    "harvest basket": "the operator bundle. The operator decides what goes to market. The harness never publishes.",
}

STATUS_MARKS = {"pass": "✅", "fail": "❌", "skipped": "⏭", "running": "⏳", "escalated": "⬆"}


def stage_label(number: int, real_name: str, style: Style) -> str:
    """`3 full model` plain, `🌳 3 trunk · full model` with emoji. The real name is always there."""
    if not style.emoji:
        return f"{number} {real_name}"
    e = STAGES[number]
    return f"{e.emoji} {number} {e.name} · {real_name}"


def closing_line(state: str, style: Style, *, where: str) -> str | None:
    """The last line of a finished run, or None while a run is still going."""
    if state == "ready-for-operator-review":
        return f"{style.icon('🧺')}ripe: the basket is at {where} (ready-for-operator-review)"
    if state == "aborted":
        return f"{style.icon('🍂')}fallen: the run was aborted (aborted)"
    return None


def frost_line(reason: str, style: Style) -> str:
    """The closing line of a run that could not finish. `reason` is the named blocking reason."""
    return f"{style.icon('❄️')}frost: {reason}; the bundle says what stopped it"


def refused_line(reasons: list[str], style: Style) -> str:
    """The closing line when the preflight blocks a run, before anything has started."""
    return f"{style.icon('❄️')}frost before planting: {', '.join(reasons)}. Nothing was started."
