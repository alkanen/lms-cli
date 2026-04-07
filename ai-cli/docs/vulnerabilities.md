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

**Status:** ✅ Fixed — `get_messages()` now checks `role not in _VALID_ROLES` and skips
the entry with a `logger.warning`, consistent with how other malformed entries are handled.
Both `add_message()` and `add_raw_message()` also validate role on write.

### Description (historical)

`get_messages()` validated that `role` and `content` were strings but did not
check whether `role` was one of the values in `_VALID_ROLES` (`system`, `user`,
`assistant`, `tool`). A history file that was manually edited or written by an
older version of the code could contain an unexpected role value that would
reach `LLMClient.send()` and cause an opaque API error.

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

## VULN-005 — Absolute-pattern check is POSIX-only in `find_files`

**Component:** `ai_cli/tools/find_files.py` — `FindFilesTool.execute()`

**Severity:** Low (Windows-only; UX/input-validation issue, not a workspace escape)

**Status:** Deferred — Windows support is currently low priority

### Description

The absolute-path guard in `execute()` rejects patterns that start with `"/"`:

```python
if pattern.startswith("/"):
    return self._err("invalid_input", "Pattern must not be an absolute path.", 400)
```

On Windows, absolute paths can also begin with a drive letter (`C:\...`) or a
UNC prefix (`\\server\share\...`). Such patterns would pass this check without
returning a clear error to the caller.

**Important:** This is a validation and UX consistency issue, not a
workspace-escape vulnerability.  The filesystem walk is always rooted at
the workspace root (a resolved absolute path).  The `pattern` argument is
only used as a match filter against relative paths and cannot influence where
`os.walk` traverses.  A Windows-style absolute pattern would simply never
match any relative path and would silently return zero results instead of a
clear `invalid_input` error.

### Conditions required

- The application must be running on Windows.
- A caller (or the LLM) must supply a drive-letter or UNC pattern.

### Proposed mitigation

Replace `pattern.startswith("/")` with an OS-aware check so Windows-style
absolute patterns also produce a clear error:

```python
import os
if os.path.isabs(pattern) or (len(pattern) >= 2 and pattern[1] == ":"):
    return self._err("invalid_input", "Pattern must not be an absolute path.", 400)
```

Or use `pathlib.PurePosixPath` / `pathlib.PureWindowsPath` to detect absolute
paths in a platform-independent way.

---

## VULN-007 — `find_files` literal-prefix narrowing follows symlinks outside the workspace

**Component:** `ai_cli/tools/find_files.py` — `FindFilesTool.execute()`

**Severity:** Low (requires a symlink to exist inside the workspace; read-only information
disclosure, no write risk)

**Status:** Deferred — symlinks inside the workspace are an explicit user action

### Description

When a glob pattern has a leading literal directory segment (e.g. `src/**/*.py`),
`find_files` narrows the `os.walk` root to that subdirectory (`walk_root = candidate`)
after confirming the directory exists (`candidate.is_dir()`).  If `candidate` is a
symlink to a directory outside the workspace, `os.walk` will traverse the symlink
target and return paths that resolve outside `workspace_root`.  The returned paths are
reported as workspace-relative strings, so the caller may not realise they originate
outside the workspace.

The same issue applies to the non-narrowed walk path: any symlinked directory that
survives the `is_ignored()` pruning step will be traversed by `os.walk`.

### Conditions required

- A symlink to an out-of-workspace directory must exist inside the workspace.
- Creating such a symlink requires write access to the workspace — it is an explicit
  user action, not something an untrusted party can trigger remotely.
- The risk is read-only information disclosure (file paths enumerated); no files are
  written or executed.

### Proposed mitigation

Before assigning `walk_root = candidate`, verify that `candidate.resolve()` is still
contained within `workspace_root.resolve()` (e.g. via `Workspace.resolve()` or a
simple `Path.is_relative_to()` check).  Alternatively, pass `followlinks=False` to
`os.walk` (the default) and skip any `dirpath` whose resolved path escapes the
workspace root.

---

## VULN-008 — `_args_summary()` may log short string argument values verbatim

**Component:** `ai_cli/core/tool_registry.py` — `_args_summary()`

**Severity:** Low (only affects the local session log; no network exposure)

**Status:** Deferred — no sensitive arguments exist in current tools

### Description

`_args_summary()` is the fallback log summary used when a tool does not
override `execute_log()`.  It logs string argument values verbatim whenever
their length is at or below `_LOG_STR_LIMIT` (currently 80 chars).  For
tools whose arguments may contain file content, prompts, or other
user-supplied text (e.g. a future tool that accepts passwords or tokens),
short values would appear in plain text in `session.log`.

Tools that handle potentially sensitive arguments should override
`execute_log()` to emit only safe metadata (e.g. path + byte count) and
never the raw value.  The `write_file` tool already does this.

### Conditions required

- A tool must accept a string argument whose value could be sensitive.
- The value must be ≤ 80 characters (longer values are already redacted).
- The attacker must have read access to `session.log` in the session
  directory (typically `~/.ai-cli/sessions/<id>/session.log`).

### Proposed mitigation

Either (a) always redact known-sensitive argument names (e.g. `content`,
`prompt`, `messages`) in `_args_summary()` regardless of length, or (b)
change the default to never log raw string values and rely entirely on
per-tool `execute_log()` overrides for safe detail.

---

## VULN-009 — `SQLiteVectorStore.search()` loads all matching vectors into memory

**Component:** `ai_cli/core/vector_store.py` — `SQLiteVectorStore.search()`

**Severity:** Low (local single-user tool; no security boundary crossed)

**Status:** Deferred — index sizes are small enough that this is not yet a practical concern

### Description

`search()` fetches all rows matching the `chunk_type`/`path_glob` filters with
`fetchall()` and deserialises every vector blob before selecting the top-k.
This is O(N · dim) in memory, where N is the total number of matching chunks
and dim is the embedding dimension (typically 768–4096 floats).

For a corpus with tens of thousands of chunks the peak allocation can reach
hundreds of MB per search call.  A sufficiently large index (or a very broad
`path_glob`) could cause the process to exhaust available memory or become
noticeably slow.

### Conditions required

- The embedding index must contain a large number of chunks (tens of thousands
  or more).
- A search query must match a large fraction of those chunks (no narrow filter).

### Proposed mitigation

Stream rows in batches using `cursor.fetchmany()` and maintain a fixed-size
min-heap of the top-k candidates so that only k entries are held in memory at
any time, reducing peak allocation from O(N · dim) to O(k · dim):

```python
import heapq
batch_size = max(128, min(4096, k * 4))
top_k = []
with self._lock:
    cursor = self._conn.execute(sql, params)
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        for row in rows:
            arr = np.frombuffer(row[-1], dtype=np.float32)
            score = float(np.dot(arr, q))
            item = (score, row[0], arr.copy(), row[1:-1])
            if len(top_k) < k:
                heapq.heappush(top_k, item)
            elif score > top_k[0][0]:
                heapq.heapreplace(top_k, item)
```

---

## VULN-010 — Runtime `/tools` commands do not affect sub-agent tool registries

**Component:** `ai_cli/core/agent.py` — `build_agent_tool_registry()`

**Severity:** Low (requires the user to disallow a tool mid-session and then invoke a sub-agent)

**Status:** Deferred — intentional design; a future `/agent` command will address per-agent runtime overrides

### Description

Sub-agents build their `ToolRegistry` at instantiation time by calling
`registry.apply_config()` against the startup-merged config.  Runtime commands
(`/tools allow`, `/tools disallow`, `/tools enable`, `/tools disable`) update
the coordinator's `ToolRegistry` in memory and persist the change to config, but
do not propagate to any sub-agent registry that was already built — nor to
sub-agent registries built after the command, because `apply_config()` re-reads
only the startup-merged config snapshot held by `ConfigManager`.

Consequently, if a user runs `/tools disallow bash` mid-session and then
triggers a sub-agent that lists `bash` in its spec, the sub-agent will still
have `bash` enabled and allowed.

### Conditions required

- The user must explicitly run a `/tools disallow` (or similar) command after
  the session has started.
- A sub-agent whose spec lists the disallowed tool must be invoked after that
  command.
- The agent spec must declare the tool — sub-agents cannot use tools not in
  their spec.

### Proposed mitigation

Introduce a `/agent` command (planned for a future PR) that allows the user to
override tool settings per agent type by name, giving explicit runtime control
over individual sub-agent capabilities without relying on the coordinator's
global `/tools` state.

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

## VULN-006 — `IgnoreFilter` implements only a subset of full `.gitignore` syntax

**Component:** `ai_cli/utils/ignore_filter.py` — `IgnoreFilter`

**Severity:** Very Low (cosmetic mismatch; no security impact)

**Status:** Deferred

### Description

`Workspace` now reads `.gitignore` in addition to `.ai-cli/.ignore`, but
`IgnoreFilter` implements a simplified subset of the full `.gitignore`
specification. Known gaps:

- **Trailing-space escaping** — `.gitignore` allows a trailing space to be
  included in a pattern by escaping it with `\ ` (backslash-space). Unescaped
  trailing spaces are stripped; escaped ones are not handled and the backslash
  is left in the pattern.
- **Character ranges in brackets** — `fnmatch` handles `[a-z]` but its
  behaviour for collating sequences may differ from Git's C-locale comparison.
- **Re-include semantics for ancestor directories** — unlike Git, this
  implementation *allows* negation to re-include a file even if its parent
  directory was matched by an earlier ignore pattern. This is intentional and
  documented, but means some `.gitignore` files will behave differently here.

### Conditions required

- A `.gitignore` file must use one of the unsupported constructs.
- Effects are limited to incorrect include/exclude decisions for affected paths;
  no data is corrupted and no security boundary is crossed.

### Proposed mitigation

Extend `IgnoreFilter` to cover the missing constructs as they are encountered
in real `.gitignore` files, or replace the hand-rolled parser with a library
that provides full `.gitignore` compatibility (e.g. `pathspec`).

---

*Add new entries above this line. Keep entries sorted by severity (High → Low).*
