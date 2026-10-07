"""The pretty status page for a person at a terminal (`status --style pretty`, or `auto` on a terminal).

A pure function of the facts dict `status.collect` returns, so it takes no lock and reads nothing. The
plain view in `status.render` is what the operator runbook and a small local model read; it is not
changed by this module. Left bar and bottom bar only (see orchard/ui.py).
"""
from __future__ import annotations

import textwrap

from orchard import lexicon
from orchard.ui import Style, visible_width

WIDTH = 80
BAR = "║  "


LABEL = 8   # columns the label takes, padded by what a person sees (an escape code takes none)


def _label(label: str, style: Style, role: str) -> str:
    """The label, coloured when it has text, padded to LABEL visible columns."""
    shown = style.paint(label, role) if label else ""
    return shown + " " * max(LABEL - visible_width(label), 0)


def _wrap(label: str, text: str, style: Style, role: str = "dim") -> list[str]:
    """`║  label  text`, wrapped so no line passes WIDTH columns; continuation lines align under the text."""
    indent = len(BAR) + LABEL
    lines = textwrap.wrap(text, WIDTH - indent, break_long_words=True, break_on_hyphens=False) or [""]
    out = [BAR + _label(label, style, role) + lines[0]]
    out += [" " * indent + ln for ln in lines[1:]]
    return out


def render_pretty(f: dict, style: Style) -> str:
    st = lexicon.STATES[f["state"]]
    sup = f["supervisor"]
    model = f["model"] or "no model yet"
    out = [f"╔══ {style.icon('🍎')}" + style.paint("tt-orchard", "title") + f" · {model}"]
    out.append(BAR + _label("state", style, "dim")
               + f"{style.icon(st.emoji)}{style.paint(st.phrase, st.role)} ({f['state']})")
    if sup["pid"] is None:
        who = "no orchardist on duty"
    else:
        who = f"orchardist pid {sup['pid']} {'on duty' if sup['alive'] else 'not alive'}"
    out += _wrap("by", who, style)
    out += _wrap("run", f["run_name"], style)
    stage = f["stage"]
    out += _wrap("stage", f"{stage['current']} {stage['name']}, attempt {stage['attempt']}"
                 if stage["current"] is not None else "none left", style)
    for r in f["stages"]:
        mark = lexicon.STATUS_MARKS.get(r["status"], "") if style.emoji else ""
        label = lexicon.stage_label(r["stage"], r["name"], style)
        extra = f", {r['attempts']} attempts" if r["attempts"] > 1 else ""
        out += _wrap("rows" if r is f["stages"][0] else "", f"{label}: {mark + ' ' if mark else ''}"
                     f"{r['status']}, {_hms(r['wall_s'])}{extra}", style)
    c = f["counts"]
    out += _wrap("counts", f"retries {c['retries']}, escalations {c['escalations']}, nudges {c['nudges']}, "
                 f"pauses {c['pauses']}, operator commands {c['operator_commands']}", style)
    if f["pause"]:
        p = f["pause"]
        out += _wrap("paused", f"stage {p['stage']} by {p['detector'] or p['kind']}: {p['reason']}", style, "warn")
    if f["control_pending"]:
        out += _wrap("control", f"{f['control_pending']} (waiting for the supervisor to read it)", style)
    d = f["disk"]
    out += _wrap("disk", f"run dir {d['run_dir_free_gb']} GB free, home {d['home_free_gb']} GB free", style)
    for i, lease in enumerate(f["leases"]):
        out += _wrap(f"{style.icon('🚪')}gate" if i == 0 else "", lease, style)
    if f["last_events"]:
        for i, e in enumerate(f["last_events"]):
            out += _wrap("events" if i == 0 else "", f"{e['seq']} {e['ts'][11:19]} {e['event']}: {e['summary']}", style)
    out += _wrap("next", f["hint"], style, "accent")
    closing = lexicon.closing_line(f["state"], style, where=f["run_dir"])
    if closing:
        out.append(f"{BAR}")
        out += _wrap("", closing, style, "good")
    out.append("╚══")
    return "\n".join(out)


def _hms(seconds: int) -> str:
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h{rem // 60:02d}m" if h else f"{rem // 60}m{rem % 60:02d}s"
