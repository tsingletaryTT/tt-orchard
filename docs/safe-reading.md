# Safe reading of files other programs wrote

`orchard/safeio.py` holds one reviewed version of each guard that a reader of an outside file needs. This page says when to use each
function, what it guarantees, what it does not cover, and how to review a new reader. The module uses only the standard library, so
the supervisor keeps running with no third-party package.

An outside file is any file or record that something else produced: a model or an agent in a run directory, an earlier run, a download,
a lab machine, another process. Reviews of this repository kept finding narrower versions of the same defects in code that reads such
files: JSON of the wrong type, NaN or Infinity, negative or fractional counts, repeated keys, a directory or file swapped for a link
after a check, a path opened by name after it was checked, a read with no size limit, a safetensors header whose tensors overlap or
run past the data, an atomic write that can be raced, and a limit that was hit without the result saying so. Each guard now lives in
one place with its own tests.

## Which function to use

| You have | Use | It gives you |
| --- | --- | --- |
| A path to a JSON file | `read_json(path, max_bytes=..., expect=dict)` | the parsed value, or one `SafeReadError` |
| A path to a file inside a directory you already hold open | `read_json(rel, anchor=pin.fd, ...)` or `pin.read_json(parts, ...)` | the same, with every path component opened without following a link |
| JSON text or bytes already in memory (a ledger line, an HTTP body) | `parse_json_bytes(data, max_bytes=...)` | the same guards without the file |
| The raw bytes of a small file | `read_bytes(path, max_bytes=...)` | the bytes, never more than the cap |
| A directory to walk | `pinned_root(dir)` then `scan_beneath(pin)` | the entries, plus a result that says whether a limit was hit |
| A path made of parts you got from outside | `open_beneath(root_fd, parts)` | a descriptor, or a refusal of `..`, empty, absolute and linked parts |
| A file to write that a reader must never see half written | `atomic_write(path, data)` | all or nothing, with the temporary file removed on every failure |
| One JSON line to add to a log under a lock you hold | `append_line(fd, line)` | one write, synced, cut back if it fails |
| A number, count or text taken from JSON | `finite_number`, `whole_number`, `bounded_str` | the value, or `BadValue` |
| A safetensors file | `safetensors_header(path_or_fd, max_header_bytes=...)` | the validated tensor table |

All the read functions raise only `SafeReadError` or a subclass: `ReadFailed` (the operating system said no), `PathRefused` (a link, a `..`,
a bad part), `NotRegular`, `TooLarge`, `ChangedWhileReading`, `NotText`, `NotJson`, `WrongType`, `BadValue`, `BadHeader`, `IncompleteScan`.
Each carries a `reason` that never holds the file's content. A reader maps these to its own documented error or its own
"missing" answer; it does not let them escape unless its callers expect them.

## What each guarantee means

**`read_bytes` and `read_json`.** The file is opened with `O_NOFOLLOW | O_NONBLOCK`. The descriptor that was opened is also the descriptor that is `fstat`-ed, and it
must be a regular file, so a FIFO does not block the open and a device such as `/dev/zero` is refused before any read. At most
`max_bytes + 1` bytes are ever requested. A larger file is `TooLarge` and is never parsed in part. The size and modification time are compared
before and after the read, so a file that grew, shrank or was rewritten while it was read is `ChangedWhileReading`. The JSON is decoded as strict
UTF-8, a byte order mark is refused, `NaN`, `Infinity` and `-Infinity` are refused, a number that overflows a float (`1e999`) is refused, an object
that repeats a key is refused, a lone surrogate in a string or key is refused, nesting deeper than `max_depth` (64) is refused, an integer of
more than `max_int_digits` (100) digits is refused, and `RecursionError` from a hostile file becomes `NotJson`. `expect` is a type or tuple of
types for the top-level value, and `true` does not count as an integer.

**`pinned_root`, `open_beneath`, `anchor`.** A directory opened once and held by descriptor is the anchor. Each component below it is opened with
`O_DIRECTORY | O_NOFOLLOW` relative to the descriptor before it, so a directory swapped for a link after discovery is refused and a root that was
renamed after the pin is not followed. Use this whenever you list a directory and then open what you found: list with `scan_beneath`, open with
`pin.read_json(entry.parts)`.

**`scan_beneath`.** The walk streams (`os.scandir` on a descriptor) and never holds a whole directory. Links are listed with kind `link` and never entered.
One entry budget is shared by every directory in the walk. A directory below the depth limit is not entered. A directory that cannot be opened
or listed is recorded. The result has `entries_cut`, `depth_cut`, `unreadable` and `incomplete`; call `require_complete()` when a partial answer would
be wrong, or report `incomplete` in your own result when it would only be less useful. A caller that reports a total from a scan must say when the scan was incomplete.

**`atomic_write`.** A temporary file in the same directory, exclusive create, full write (short writes are retried), exact mode, `fsync`, rename
(`os.replace`, or `os.link` plus unlink when `replace=False`, which fails and leaves the old file if the target exists), then `fsync` of the directory.
The temporary file is removed on every failure path. A link at the target is replaced and never written through.

**`append_line`.** The caller opens the file with `O_RDWR | O_APPEND`, takes the lock, and closes both afterwards. The function writes the line and a
newline with one `write` loop, `fsync`s, and on failure cuts the file back to its earlier length. `heal_torn=True` puts a newline before the line when the last
line has no newline. `orchard/ledger.py` and `orchard/telegraph.py` use the same pattern with different lock and torn-tail policies; the function covers
the shared part and leaves those two policies to the caller.

**`safetensors_header`.** The 8-byte length and the JSON header are read from one descriptor with `pread`. The length must be at least 2, within
`max_header_bytes`, and fit in the file. Every entry must have exactly `dtype` (a known one), `shape` (up to 64 non-negative whole numbers, element
count at most 2**63 - 1) and `data_offsets` (two whole numbers with `0 <= start <= end <= data size`), and `end - start` must equal the element count
times the dtype size. Ranges may not overlap. With `contiguous=True` (the default) they must also tile the data section with no gap. `__metadata__` must
be an object of strings. A repeated tensor name is refused by the JSON parser.

## What these functions do not cover

* A path typed by an operator or read from a config cannot be anchored: there is no trusted directory to anchor it to. `O_NOFOLLOW` guards only the last
  component of a path opened by name, so a link in an earlier component is followed. Use `pinned_root` for a directory you walk, and accept the
  residual risk for a single path an operator named.
* Nothing here stops a user who already controls the process or the directory in which the file lives. The guards address files that other programs,
  earlier runs, models and downloads wrote.
* A file that grows after the size check is read only up to `max_bytes + 1` bytes, and the read then fails as `TooLarge` or `ChangedWhileReading`. A writer
  that replaces bytes without changing the size and modification time is not seen. Files that report a size of zero but have content (files under `/proc`)
  are refused as `ChangedWhileReading`.
* `read_json` returns valid JSON as it is. Whether `{}`, `[]`, `null`, `0` or `""` is acceptable, and what a field means, is the caller's decision.
  Pass `expect`, and check each field with `whole_number`, `finite_number` and `bounded_str`; do not use `isinstance(x, int)` alone, which accepts `True`.
* `atomic_write` does not stop two writers from both succeeding (the last rename wins). It does not make a write durable on a file system that ignores `fsync`.
* `append_line` takes no lock and recovers no torn tail. `scan_beneath` does not read files; a file created after its directory was listed is not seen.
* `safetensors_header` does not read tensor data, so it says nothing about the values. A dtype the format adds later is refused until `DTYPE_BITS` is updated.
* A file read as `NotRegular` or `PathRefused` may be a legitimate setup (a link to a shared cache). Decide in the reader whether that is an error.

## Reviewing a new reader

Ask these of any code that reads a file or a record that something else wrote. Each question has a function above that answers it.

1. Is the path opened by name after it was checked (`exists()`, `is_file()`, `resolve()`, `stat()`)? Open once, check the descriptor.
2. Can the last component be a link, and can an earlier component be swapped after you listed the directory? Use `anchor` or `pinned_root`.
3. Is there a size limit before the read, and is a larger file an error, with no partial parse?
4. Can the path be a FIFO, a device, a directory, a dangling link or a link loop? Does the reader finish and name the failure?
5. Which exceptions can leave the function (`RecursionError`, `ValueError` from a long integer, `UnicodeDecodeError`, `OSError`, `AttributeError`) and does the docstring say so?
6. Is the top-level type checked before any `.get` or index? Is each field checked for the type it needs, with bool excluded from numbers and floats excluded from counts?
7. Can a number be NaN, Infinity or out of range, and can a count be negative or fractional? Can a string hold a lone surrogate or be very long?
8. Can an object repeat a key, so that the last value hides the first?
9. When a limit is hit (entries, depth, bytes, lines), does the result say so, so a caller cannot report a complete answer?
10. Does a write go through a temporary file in the same directory, with the file and the directory synced, and is the temporary file removed on failure?
11. Is the reader in the hostile-input registry? A reader with no entry has not been shown to survive the corpus.

## Registering a reader and the backlog

`tests/hostile_inputs.py` holds the corpus (`BYTES_CORPUS`, `ITEMS`), the registry (`READERS`) and the child-process runner. Add a `Reader` with a function
that takes a base directory (or the payload bytes) and runs your reader. State `documented` (the exception types that are part of its contract),
`rel` (where the hostile file goes beneath the base directory), `suffix=b"\n"` for a line-oriented reader, and `setup` to build the valid files around it.
`tests/test_hostile_readers.py` then runs every item against it in a child process with a 2 GiB address-space limit.

A reader that fails an item today is not weakened. File a public GitHub issue with the `fidelity` label (no machine paths, host names or tokens), and list the
pair in `KNOWN_DEFECTS` under that issue number. The test is then `xfail(strict=True)`, and fixing the reader turns it red until the entry is removed.
`UNREGISTERED_MODULES` lists the modules that parse JSON and have no reader registered yet; a test keeps the list exact, so a new module that parses JSON
fails until it is registered.

## Moving a reader onto the module

Before:

```python
try:
    rec = json.loads(path.read_text(encoding="utf-8"))
except (OSError, ValueError):
    return None
return rec if isinstance(rec, dict) else None
```

After:

```python
try:
    return safeio.read_json(path, max_bytes=64 * 1024, expect=dict)
except safeio.SafeReadError:
    return None
```

The cap is the largest file the writer could legitimately produce, with room to spare. The reader keeps its own meaning for a missing or bad file. Then remove the
reader's entries from `KNOWN_DEFECTS`; the strict `xfail` marks tell you which.
