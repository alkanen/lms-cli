# Plan: bash tool

> Source PRD: GitHub Issue #15 — Tool: Add `bash` tool

## Architectural decisions

- **Tool class**: `BashTool` in `ai_cli/tools/bash.py`, subclasses `Tool` from `ai_cli/tools/base.py`
- **Class attributes**: `NAME = "bash"`, `PERMISSION_REQUIRED = True`, `DISABLED_BY_DEFAULT = True`
- **Execution (Phases 1–3)**: `subprocess.run` without `shell=True`; command is parsed internally via `shlex` and executed directly to avoid shell injection. `cwd` is set to the workspace root and `stdin` to `DEVNULL` for deterministic, non-interactive execution.
- **Execution (Phase 4+)**: chained commands (`|`, `&&`, `||`) require shell semantics and will use `shell=True` with the original command string. Permission checking is done on the parsed segments before execution; the shell injection risk is mitigated by requiring explicit user approval for every segment.
- **Grant storage**: session-scoped in-memory lists on the tool instance (`_exact_grants: set[str]`, `_pattern_grants: list[str]`), cleared by `reset_session_state()`
- **Grant matching**: incoming segment is `shlex.split` → rejoined as normalised string, then checked against stored exact grants (set membership) and pattern grants (`fnmatch.fnmatch`)
- **Env var stripping**: `KEY=val` prefixes are parsed off before building permission options and matching grants; applied to subprocess via `env=` kwarg
- **Permission option shape** (per non-heredoc segment):
  - Universal: `yes`, `no`, `always` (exact match), `custom`
  - Extra: `always: <executable> <leading args> *` (wildcard trailing args)
- **Result shape**:
  - `capture=stdout/stderr/interleaved` → `{"status": "success", "data": {"output": "..."}}`
  - `capture=separate` → `{"status": "success", "data": {"stdout": "...", "stderr": "..."}}`
  - Truncation appends `"warning": "Output truncated at N bytes"` to `data`
- **Tests**: `tests/test_bash.py`, mirroring the structure of `tests/test_write_file.py`

---

## Phase 1: Core tool — single command + permission grants

**User stories**: basic command execution; exact and wildcard-trailing "always" grants; session-scoped grant store; `shlex` normalisation and `fnmatch` matching.

### What to build

A `BashTool` that accepts a single `command` string, executes it via subprocess, and returns captured stdout. The tool manages a session-scoped grant store. Before executing, it checks stored grants (exact match first, then fnmatch patterns). If no grant matches, it prompts the user with the four universal options plus one extra: `always: <executable> <leading args> *`. The universal `always` stores an exact grant; the extra option stores a wildcard pattern. `reset_session_state()` clears both stores.

The tool is disabled by default (like `write_file`) and requires permission on every call unless auto-granted.

### Acceptance criteria

- [ ] `BashTool` registers in the tool registry with `NAME="bash"`, `PERMISSION_REQUIRED=True`, `DISABLED_BY_DEFAULT=True`
- [ ] `definition()` declares a single required `command` string argument
- [ ] A simple command (e.g. `echo hello`) executes and returns `{"status": "success", "data": {"output": "hello\n"}}`
- [ ] Permission is requested before execution; denial returns `{"status": "error", "error": "permission_denied", ...}`
- [ ] `extra_permission_options()` returns one extra option: `always: <executable> <leading args> *`
- [ ] Selecting `always` (universal) stores the normalised full command as an exact grant; subsequent identical calls skip the prompt
- [ ] Selecting the extra wildcard option stores a pattern; subsequent calls matching that pattern skip the prompt
- [ ] `reset_session_state()` clears both grant stores
- [ ] `execute_log()` returns a compact summary (first 60 chars of command)
- [ ] Tests cover: schema, basic execution, permission denial, exact grant, wildcard grant, grant cleared on reset

---

## Phase 2: Output capture and truncation

**User stories**: `capture` enum arg; `max_output_bytes` arg; truncation warning; separate stdout/stderr result.

### What to build

Add two optional arguments to the schema: `capture` (enum: `stdout`, `stderr`, `interleaved`, `separate`; default `stdout`) and `max_output_bytes` (integer; default 1024). Wire subprocess output accordingly. For `separate`, return `{"stdout": "...", "stderr": "..."}` in `data`; for all others return `{"output": "..."}`. When captured output exceeds `max_output_bytes`, truncate and add `"warning": "Output truncated at N bytes"` to `data`.

### Acceptance criteria

- [ ] `capture=stdout` (default) returns only stdout in `output`
- [ ] `capture=stderr` returns only stderr in `output`
- [ ] `capture=interleaved` merges stderr into stdout in `output`
- [ ] `capture=separate` returns both `stdout` and `stderr` fields
- [ ] Output exceeding `max_output_bytes` is truncated and `warning` field is present
- [ ] Output within `max_output_bytes` has no `warning` field
- [ ] `max_output_bytes` defaults to 1024 when omitted
- [ ] Tests cover all four capture modes, truncation boundary, no-warning path

---

## Phase 3: Environment variable support

**User stories**: `KEY=val` prefixes applied to subprocess env; stripped from permission prompt and grant matching.

### What to build

Before building the permission prompt and before matching grants, strip any leading `KEY=val` tokens from the command (using `shlex` to detect the pattern `[A-Z_][A-Z0-9_]*=.*`). Apply the stripped vars to the subprocess `env`. The full command string including env vars is shown in the human-readable permission question so the user can see them, but the grant key/pattern is built from the env-stripped remainder only.

### Acceptance criteria

- [ ] `MYVAR=123 python3 -c "import os; print(os.environ['MYVAR'])"` prints `123` to stdout
- [ ] The permission question shows the full command including `MYVAR=123`
- [ ] The extra permission option (`always: python3 *`) does NOT include the env var prefix
- [ ] A grant stored for `python3 *` matches a future call with a different env var prefix
- [ ] Env vars are isolated to the subprocess and do not leak into the parent process environment
- [ ] Tests cover: env var application, stripping from grant key, cross-prefix grant matching

---

## Phase 4: Chained commands

**User stories**: `|`, `&&`, `||` split into segments; chain summary header; per-segment sequential permission; whole-chain denial on rejection; one-time grants consumed; always grants recorded.

### What to build

Parse the command string to identify chain operators (`|`, `&&`, `||`) and split into segments. Before any permission prompts, display a summary header listing all executables with their chain operators but with arguments stripped (e.g. `cat | grep | awk`). Then iterate segments: skip those already covered by a stored grant; prompt for the rest in order. If any segment is denied, return a permission-denied error immediately — the chain is not executed. One-time (`yes`) grants from earlier segments are considered spent. `always` grants (exact or pattern) from any segment are recorded to the store regardless of whether a later segment is denied.

For execution, the chained command is passed to the shell (via `shell=True` with the full command string) so that operators work correctly. Permission checking is done on the parsed segments; execution uses the original string.

### Acceptance criteria

- [ ] A pipe chain prompts for each unapproved segment sequentially
- [ ] The chain summary header shows executables + operators with args stripped
- [ ] Approving all segments executes the full chain and returns output
- [ ] Denying any segment returns `permission_denied`; no execution occurs
- [ ] A one-time grant for segment 1 does not re-prompt on the next (different) call
- [ ] An `always` grant recorded before a later denial is present in the store after the call
- [ ] `&&` and `||` chains are parsed and checked the same way as `|`
- [ ] Tests cover: all-approved pipe, mid-chain denial, always grant survives denial, `&&` and `||`

---

## Phase 5: Redirection segments

**User stories**: `>`, `>>`, `2>&1`, `>&2`, `< file` parsed as independent permission segments; each has exact + wildcard grant options; path-pattern grants supported.

### What to build

Extend the command parser to extract redirection tokens (`>`, `>>`, `2>`, `2>&1`, `>&2`, `<`, `<<`) and their targets as separate segments inserted after the command segment they belong to. Each redirection segment gets its own permission prompt with the full redirection token as the display string and two grant options: exact match and a wildcard pattern (e.g. `> ./docs/*`). Grant matching for redirections uses the same `fnmatch` logic as command segments.

### Acceptance criteria

- [ ] `ls path 2>&1` results in two permission checks: `ls path` and `2>&1`
- [ ] `cat file > output.txt` results in two checks: `cat file` and `> output.txt`
- [ ] A grant of `> ./docs/*` auto-approves any future `> ./docs/<anything>`
- [ ] Denying the redirection segment denies the whole chain
- [ ] `>>` (append) is treated as a distinct operator from `>` for matching purposes
- [ ] Tests cover: stdout redirect, stderr redirect, append, wildcard path grant, denial

---

## Phase 6: Heredoc support

**User stories**: `<<EOF...EOF` detected; only one-time permission offered.

### What to build

Detect heredoc syntax (`<<MARKER` ... `MARKER`) in the command string. When a heredoc is present anywhere in the command, the permission prompt for that segment offers only the universal `yes`, `no`, and `custom` options — the `always` option and the wildcard extra option are suppressed. This is enforced in `extra_permission_options()` by returning an empty list for heredoc-containing segments, and by overriding `request_permission()` to exclude `always` from the universal choices passed to the prompt.

### Acceptance criteria

- [ ] A command with `<<EOF` executes correctly and captures the heredoc content
- [ ] The permission prompt for a heredoc segment shows only `yes`, `no`, `custom` — no `always`, no wildcard extra option
- [ ] Selecting `yes` executes once; the next identical call prompts again
- [ ] A non-heredoc segment in the same chain still offers `always` options
- [ ] Tests cover: heredoc execution, no always option offered, re-prompt on identical call, mixed heredoc + normal segment chain

---

## Phase 8: Process substitution

**User stories**: `<(cmd)` and `>(cmd)` syntax detected; inner command receives its own permission check; bash invoked explicitly.

### What to build

Detect bash process substitution (`<(...)` and `>(...)`) in command segments. When present, extract each inner command and run it through the same permission flow as a top-level command — the inner command is checked before the outer command executes. Denying any inner command denies the whole call. Because process substitution is bash-specific (not POSIX sh), the tool must invoke `bash -c` rather than `sh -c` (or the system default shell) whenever a process substitution is detected. Grant matching for inner commands uses the same exact + fnmatch logic as regular command segments; grants stored for inner commands are session-scoped and cleared by `reset_session_state()`.

### Acceptance criteria

- [ ] `base64 -i <(echo -n "test")` executes correctly and returns `dGVzdA==`
- [ ] The inner `echo -n "test"` receives its own permission prompt before the outer command runs
- [ ] Denying the inner command returns `permission_denied`; the outer command is not executed
- [ ] An `always` grant for the inner command auto-approves it on subsequent calls
- [ ] `>(cmd)` output process substitution is detected and checked the same way as `<(cmd)`
- [ ] When process substitution is present the tool invokes `bash`, not `sh`
- [ ] Commands without process substitution are unaffected (no shell change)
- [ ] Tests cover: basic `<(cmd)` execution, inner command denial, always grant for inner command, `>(cmd)`, no regression on plain commands
