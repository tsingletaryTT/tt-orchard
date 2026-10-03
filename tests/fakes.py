"""Test doubles shared by the plan 3 tests. Nothing here touches hardware or the network."""
from __future__ import annotations

import json

from orchard.adapters import Lease
from orchard.commands import CommandResult

OWNER = 4242

# Shape copied from a real grant: runs/h5-20261002T201323Z/lease.json (owner pid changed).
GRANT = {"chips": ["0000:01:00.0", "0000:02:00.0"], "dev_indices": [0, 1],
         "env": {"TT_VISIBLE_DEVICES": "0000:01:00.0,0000:02:00.0"}, "expanded": True,
         "granted": True, "lease_id": "fb9995", "neighbours": {}, "owner_pid": OWNER,
         "requested": 1, "units": ["0000046131924062"]}

LEASE = Lease("fb9995", ("0000:01:00.0", "0000:02:00.0"), (0, 1),
              {"TT_VISIBLE_DEVICES": "0000:01:00.0,0000:02:00.0"}, ("0000046131924062",))


class FakeRun:
    """Stands in for orchard.commands.run_command.

    script maps a key to a list of answers, used in order; the last answer repeats. The key is
    argv[1] (the gozer subcommand) unless `key` is a function of argv. An answer is
    (returncode, stdout, stderr). stdout may be a dict, sent as JSON. returncode may be
    "timeout". A call with no scripted answer fails the test.
    """

    def __init__(self, script: dict | None = None, key=None):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.key = key or (lambda argv: argv[1])
        self.calls: list[dict] = []

    def __call__(self, argv, timeout, *, env=None, kill_on_timeout=True):
        argv = [str(a) for a in argv]
        self.calls.append({"argv": argv, "timeout": timeout, "env": env,
                           "kill_on_timeout": kill_on_timeout})
        answers = self.script.get(self.key(argv))
        if not answers:
            raise AssertionError(f"unexpected call: {argv}")
        rc, out, err = answers.pop(0) if len(answers) > 1 else answers[0]
        if rc == "timeout":
            return CommandResult(tuple(argv), None, timed_out=True, left_running=not kill_on_timeout)
        return CommandResult(tuple(argv), rc, out if isinstance(out, str) else json.dumps(out), err)

    def argvs(self) -> list[list[str]]:
        return [c["argv"] for c in self.calls]
