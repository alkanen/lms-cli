# Known Vulnerabilities and Security Considerations

This document records security issues that have been identified but intentionally
deferred. Each entry explains the nature of the risk, the conditions required to
exploit it, and a proposed mitigation if one is known.

---

## VULN-001 — Symlink substitution attacks on session files

**Component:** `ai_cli/core/session_manager.py`

**Severity:** Low (requires local write access to the sessions directory)

**Status:** Deferred — overkill for current threat model

### Description

Session files (`history_current.jsonl`, `history_full.jsonl`, `metadata.yaml`)
are opened by path. If an attacker can write to the sessions directory
(`~/.ai-cli/sessions/<session-id>/`) before a file operation occurs, they could
replace a session file with a symlink pointing to an arbitrary target
(e.g. `/etc/passwd`). Subsequent writes by the session manager would then land
in the symlink target rather than the intended file.

This is a classic TOCTOU (Time Of Check To Time Of Use) race. Attack windows
include:

1. **Between existence check and open** — `Path.exists()` resolves symlinks, so
   a symlink planted between the check and the `open()` call would be silently
   followed.
2. **During rollback in `compact()`** — the rollback path uses
   `Path.write_bytes()`, which is non-atomic and follows symlinks
   unconditionally. If the history file is replaced with a symlink between the
   primary write and the rollback, the rollback would write to the symlink
   target.

### Conditions required

- The attacker must have write access to the user's sessions directory.
- In normal single-user deployments the directory is owned by the current user
  (`0700`), making this scenario implausible in practice.
- The risk increases in shared-directory environments, containers with mounted
  volumes, or if another process is compromised.

### Proposed mitigation

1. **Fast-fail `is_symlink()` checks** before any file I/O on session paths.
2. **`O_NOFOLLOW` at the OS level** via `os.open()` + `os.fdopen()` so the
   kernel refuses to open a symlink even if a race occurs between the check and
   the open.
3. **Atomic rollback** in `compact()` using `tempfile.NamedTemporaryFile` +
   `Path.replace()` instead of `Path.write_bytes()`.

---

## VULN-002 — `get_messages()` does not validate role against known values

**Component:** `ai_cli/core/session_manager.py` — `Session.get_messages()`

**Severity:** Low (requires a manually edited or externally corrupted history file)

**Status:** Deferred

### Description

`get_messages()` validates that `role` and `content` are strings but does not
check whether `role` is one of the values in `_VALID_ROLES` (`system`, `user`,
`assistant`, `tool`). A history file that was manually edited or written by an
older version of the code could contain an unexpected role value. This would be
returned to callers without warning and could cause `LLMClient.send()` to fail
with an opaque API error during the next turn or during `compact()`.

### Conditions required

- The history file must contain an entry with an unrecognised `role` value.
- This requires either manual editing of the file or an old/external writer.
- Normal usage through `add_message()` prevents this (role is validated there).

### Proposed mitigation

In `get_messages()`, after confirming `role` is a string, also check
`entry["role"] not in _VALID_ROLES` and skip the entry with a `logger.warning`
if it fails, consistent with how other malformed entries are handled.

---

## VULN-004 — `get_messages()` passes through unvalidated field values for tool-call messages

**Component:** `ai_cli/core/session_manager.py` — `Session.get_messages()`

**Severity:** Low (requires a manually edited or externally corrupted history file)

**Status:** Deferred

### Description

`get_messages()` validates that `role` is a known value, and that each message
contains at least one of `content` or `tool_calls`.  However, it does not
validate the *types* of those fields per role:

- A `tool` message with a non-string `content` is passed through unchecked.
- An `assistant` tool-call message with a malformed `tool_calls` value (e.g.
  a string instead of a list) is passed through unchecked.

An invalid message shape sent to the OpenAI API will fail the entire turn with
an opaque 400 error rather than being skipped with a warning, contrary to what
the docstring promises for malformed lines.

### Conditions required

- The history file must contain an entry with a structurally invalid field
  value (e.g. `tool_calls` as a string, `content` as a number).
- This requires either manual editing of the file or an external writer.
- Normal usage through `add_message()` and `add_raw_message()` prevents this
  as the REPL only writes well-typed dicts.

### Proposed mitigation

In `get_messages()`, add per-role structural validation:
- For `role == "tool"`: require `isinstance(content, str)` and
  `isinstance(tool_call_id, str)`.
- For `role == "assistant"` with `tool_calls` present: require
  `isinstance(tool_calls, list)`.
- Skip lines that fail validation with a `logger.warning`.

---

## VULN-003 — Orphan session directory left behind on metadata write failure in `new()`

**Component:** `ai_cli/core/session_manager.py` — `SessionManager.new()`

**Severity:** Very Low (accumulates clutter but causes no data loss or error)

**Status:** Deferred

### Description

`SessionManager.new()` creates the session directory with `mkdir()` before
calling `session._write_meta(meta)`. If the metadata write fails (e.g. disk
full, permission denied), the empty session directory is left behind. `list()`
silently skips directories without `metadata.yaml`, so these orphans are
invisible to the application but accumulate in the sessions directory over time.

### Conditions required

- `_write_meta()` must fail after `mkdir()` succeeds.
- This requires an unusual filesystem condition (full disk, revoked permissions).

### Proposed mitigation

Wrap `_write_meta()` in a `try/except SessionError` block; on failure,
best-effort `shutil.rmtree(session_dir)` the newly created directory before
re-raising, so the sessions directory stays clean.

---

*Add new entries above this line. Keep entries sorted by severity (High → Low).*
