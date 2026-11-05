# Tool-result dedupe in the agent loop

An agent step that reads the same unchanged file again and again fills its context with text it already holds. Within one agent step, the
tools now answer a repeat read of an unchanged file with a short note, and the watchdog sends one nudge when several notes come close together.
The code is `Tools` in `orchard/agent.py` and `DedupeStreak` in `orchard/watchdog.py`. The option `--no-tool-dedupe` turns it off for a run.

## What the logs show

The stage logs of runs 12 to 16 of the tiny Qwen4 rehearsal (37 agent steps, counting a step and its gate-feedback and wrap-up continuations as one) hold 345 `read_file` calls that can be paired with their results.
Counting a call as a repeat when the same path and arguments came earlier in the same step, 43 of 353 calls (12 percent) were repeats. The worst
step re-read 7 files 21 times.

`scripts/dedupe_replay.py` replays the recorded calls through the rules below, read only. The logs hold no file identity, so it takes equal text
to mean an unchanged file. It follows the live loop: a record that opens a conversation (a context restart logs its replacement prompt from the start) makes
the step forget what it read, the incomplete calls of a capped reply have no result and are left out, tool arguments may be a JSON string or a dict, and the
logs of a step's continuations share its memory. A path is matched the way the live tools resolve it (the run directory first, then the stage directory),
with the stage component kept: `stages/1/evidence/a.txt` and `stages/2/evidence/a.txt` are two files, and a stage-relative path means the stage of its log.
Its result for those five runs (unchanged by that rule):

| | calls | would have been notes |
|---|---:|---:|
| `read_file` | 345 | 31 (9 percent) |
| read-only shell commands | 321 | 2 |

The notes would have kept about 293,000 of the 2,740,000 characters of tool output (10.7 percent) out of the contexts again. The largest single
step would have had 14 notes (run 12, a step that read 7 files 21 times).

This is a partial lever. Many loops use calls that differ a little each time: `cd <run directory>`, `sed -n` with another range, `find` with another
path. Those are not exact repeats, the dedupe does not touch them, and the repeated-call and turn-repeat detectors stay as they were.

## A restart and the hint

The hint says a file's text is already in the context. A context restart drops that conversation, so the hint is not carried into the replacement
prompt (it starts with the marker `DEDUPE_HINT_PREFIX`, which the agent loop checks; the other watchdog nudges are still carried), the replacement prompt says
that what the agent read before is no longer in the conversation and a file may be read again, and the tools forget every read and command they remembered, so
the first read after a restart returns the text and never a note.

## The rules

**Reads.** `Tools` keeps, for one agent step, a record of each `read_file` page: the file's real path and the offset, the file's identity
(device, inode, size and modification time in nanoseconds) taken before the read, a hash of the text returned, and the model turn. The identity is
read again from the open file after the read; a file that changed in between is not recorded. A later read of the same page answers with a note when
the identity and the hash are both unchanged:

> `<path>: unchanged since turn N; its contents are in your context above (turn N). Next step: write or edit your files; do not read it again.`

A file the agent wrote, a file replaced by another inode (even with the same text), a file whose size or time changed, and a page of a long
file at another offset all return their text. Errors and refusals are never recorded.

**Shell commands.** A command is answered with a note only when it is one of a short list of read-only forms, it ran earlier in the step with the
same command string, it exited 0, and it printed exactly what it printed then. The list: `ls`, `cat`, `head`, `tail`, `wc`, `stat`, `sed -n 'a,bp'`,
`grep` without a recursive flag, and `find` with only `-name`, `-iname`, `-type` and `-maxdepth`. Every file argument must lie inside the run
directory. A command with a separator, redirect, substitution, escape, subshell or an unquoted wildcard is off the list, and so is a command that
reads standard input. A `write_file`, and any command that is off the list, makes the step forget every command it remembered. The command still
runs each time; the note replaces only its output in the context.

**One step.** The supervisor makes new tools for each agent step, so nothing carries over from one step to the next. A step that starts again from
its task after capped replies forgets its reads too, because the conversation that held them is gone. A continuation of a step keeps its reads.

**The watchdog.** A note is a `tool_result` event with `deduped` set. The detector `dedupe_streak` fires when 3 of the last 6 tool results of a
step were notes, at most once per step, and names the file. Its nudge tells the agent to stop reading and write. It is not one of the ladder's
three rungs: it uses none of their caps, is sent at most 3 times per agent and stage, and never leads to an escalation or a pause. Past those 3 it
is a ledger notice.

**The record.** An agent step's `transcript` evidence entry in the ledger gains `tool_deduped` (the count) and `tool_deduped_calls` (up to 12
entries of tool, turn and the turn of the first read) when at least one note was given. A step without notes has the same entry as before.

## The option

`tt-orchard bringup --no-tool-dedupe` (and `python3 -m orchard.supervisor run --no-tool-dedupe`) sends the agent every repeat read in full. A run
that starts with the option off records `tool_dedupe: false` in `run_start`; a run with the default records nothing. A resume keeps the recorded
choice. A resume that turns it off for a run that started with it on is refused, because the choice changes what the agent was sent.

The kernel probe's agent keeps its own reads and is not changed.

## Not measured

No model was involved in any of this. Whether the notes shorten loops, or change what the coder does after a note, is not known; that needs a run
of the same rehearsal with the option on and off. The offline numbers above are an estimate from recorded text.
