# How to Write a Custom Tool

Custom tools let you extend ai-cli with project-specific or personal actions.
They are plain Python files dropped into a well-known directory — no package
installation required.

---

## Where to put your tool files

There are two locations, both scanned automatically at startup:

| Location | Scope |
|---|---|
| `~/.ai-cli/tools/` | Global — available in every project |
| `<project-root>/.ai-cli/tools/` | Project — only available in this project |

Load order is: bundled tools → global tools → project tools.  When a later
tier defines a tool with the same `NAME` as an earlier tier, the later one
wins (with a warning in the log).  This lets you override or replace a
bundled tool for a specific project.

---

## One file or many?

Both are fine.  The loader scans every `*.py` file in the tools directory
(skipping files whose name starts with `_`) and registers **all** `Tool`
subclasses found in each file.  You can put related tools together:

```
.ai-cli/tools/
    git_tools.py        # GitStatusTool, GitDiffTool, GitLogTool
    db_tools.py         # QueryTool, SchemaTool
    _helpers.py         # NOT loaded — leading underscore is skipped
```

---

## Minimal example

```python
"""echo_tool — repeat a message back to the caller."""

from __future__ import annotations

from ai_cli.tools.base import Tool, ToolArgument, ToolSchema


class EchoTool(Tool):
    NAME = "echo"
    DESCRIPTION = "Repeat a message back. Useful for testing custom tools."
    PERMISSION_REQUIRED = False   # True = prompt user before each call

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    name="message",
                    description="The text to echo.",
                    argument_type="string",
                    required=True,
                ),
            ],
        )

    def execute(self, *, message: str) -> dict:
        return self._ok({"echoed": message})
```

---

## Required class attributes

| Attribute | Type | Purpose |
|---|---|---|
| `NAME` | `str` | Tool identifier — used everywhere (config, `/tools`, LLM schema). Must be unique. |
| `DESCRIPTION` | `str` | One-line summary shown to the LLM and in `/tools list`. |
| `PERMISSION_REQUIRED` | `bool` | Default permission behaviour. `True` = prompt the user before each call. |

---

## Optional class attribute

| Attribute | Type | Default | Purpose |
|---|---|---|---|
| `DISABLED_BY_DEFAULT` | `bool` | `False` | Start disabled; user must explicitly `/tools enable <name>`. |

---

## Implementing `definition()`

`definition()` must return a `ToolSchema` instance describing your tool.  The
`ToolSchema` and `ToolArgument` helpers make it easy to define the underlying
OpenAI function-calling schema:

```python
def definition(self) -> ToolSchema:
    return ToolSchema(
        name=self.name,          # always self.name, not a hard-coded string
        description=self.description,
        arguments=[
            ToolArgument(
                name="path",
                description="Workspace-relative file path.",
                argument_type="string",   # "string" | "integer" | "number"
                                          # "boolean" | "array" | "object"
                required=True,
            ),
            ToolArgument(
                name="mode",
                description="Access mode.",
                argument_type="string",
                enum=["read", "write"],   # restrict to a fixed set of values
            ),
            ToolArgument(
                name="lines",
                description="List of line numbers.",
                argument_type="array",
                items={"type": "integer"},  # element type for arrays
            ),
        ],
    )
```

Arguments without `required=True` are optional — the LLM may omit them, so
your `execute()` must give them default values.

---

## Implementing `execute(**kwargs)`

`execute()` receives keyword arguments matching the names declared in
`definition()`.  Always use `*` to force keyword-only arguments:

```python
def execute(self, *, path: str, mode: str = "read") -> dict:
    ...
    return self._ok({"result": "..."})     # success
    return self._err("error_code", "Human-readable message.", 400)  # failure
```

### Result shapes

```python
# Success
{"status": "success", "data": {...}}          # via self._ok(data_dict)

# Error
{"status": "error", "error": "<code>",
 "message": "<text>", "code": <int>}         # via self._err(code, msg, http_code)
```

---

## Accessing the workspace

Your tool receives a `Workspace` instance at `self._workspace`.  Use its
methods rather than direct `pathlib`/`os` calls — they enforce ignore rules
and path-escape protection automatically:

```python
# Read a file
text = self._workspace.read_file("./src/main.py")

# Resolve to an absolute Path (raises WorkspaceError if outside root)
abs_path = self._workspace.resolve("./src/main.py")

# Project root
root = self._workspace.root   # pathlib.Path
```

---

## Permission-gated tools

Set `PERMISSION_REQUIRED = True` and call `self.request_permission()` before
performing the action:

```python
PERMISSION_REQUIRED = True

def execute(self, *, path: str) -> dict:
    allowed, _ = self.request_permission(f"Delete {path}")
    if not allowed:
        return self._err("permission_denied", "User denied.", 403)
    # ... do the action ...
    return self._ok({})
```

`request_permission()` is a no-op (returns `(True, "")`) when
`permission_required` is `False`, so the check is safe to leave in even
if the tool is later reconfigured to not require permission.

### Custom permission options

Override `extra_permission_options()` to add choices beyond the universal
four (yes / no / always / custom).  Override `on_permission_granted()` to
react when the user picks one of your custom options:

```python
def extra_permission_options(self, *, path: str, **_: object) -> list[str]:
    # offer "always allow this directory"
    return [f"dir:{path}"]

def on_permission_granted(self, choice: str, **kwargs: object) -> None:
    if choice.startswith("dir:"):
        self._always_allowed_dirs.add(choice[4:])
```

---

## Session state

If your tool accumulates state during a session (caches, allow-lists, etc.)
override `reset_session_state()` so it is cleared when the user runs
`--resume`/`--continue`:

```python
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._cache: dict[str, str] = {}

def reset_session_state(self) -> None:
    self._cache.clear()
```

---

## Full example with two related tools in one file

```python
"""fs_tools — simple filesystem helpers for ai-cli projects."""

from __future__ import annotations

from pathlib import Path

from ai_cli.core.workspace import WorkspaceError
from ai_cli.tools.base import Tool, ToolArgument, ToolSchema


class ListDirTool(Tool):
    NAME = "list_dir"
    DESCRIPTION = "List files and subdirectories inside a workspace directory."
    PERMISSION_REQUIRED = False

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    name="path",
                    description="Directory path relative to the workspace root.",
                    argument_type="string",
                    required=True,
                ),
            ],
        )

    def execute(self, *, path: str) -> dict:
        try:
            abs_path = self._workspace.resolve(path)
        except WorkspaceError as exc:
            return self._err("invalid_path", str(exc), 400)

        if not abs_path.is_dir():
            return self._err("not_a_directory", f"'{path}' is not a directory.", 400)

        entries = sorted(
            ("dir" if e.is_dir() else "file", e.name)
            for e in abs_path.iterdir()
        )
        return self._ok({"path": path, "entries": entries})


class FileStatTool(Tool):
    NAME = "file_stat"
    DESCRIPTION = "Return size and modification time for a file in the workspace."
    PERMISSION_REQUIRED = False
    DISABLED_BY_DEFAULT = True   # opt-in — user must /tools enable file_stat

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    name="path",
                    description="File path relative to the workspace root.",
                    argument_type="string",
                    required=True,
                ),
            ],
        )

    def execute(self, *, path: str) -> dict:
        try:
            abs_path = self._workspace.resolve(path)
        except WorkspaceError as exc:
            return self._err("invalid_path", str(exc), 400)

        if not abs_path.is_file():
            return self._err("not_a_file", f"'{path}' is not a file.", 400)

        stat = abs_path.stat()
        return self._ok({
            "path": path,
            "size_bytes": stat.st_size,
            "modified": stat.st_mtime,
        })
```

---

## Managing tools at runtime

Once your file is in place, use slash commands to control it:

```
/tools list                      # see all tools and their status
/tools info my_tool              # full details for one tool
/tools enable my_tool            # enable (persists to project config)
/tools disable my_tool           # disable (persists)
/tools enable --session my_tool  # enable for this session only
/tools allow my_tool             # lift a disallow (persists)
/tools disallow my_tool          # hard-block a tool (persists)
```

Or set the initial state in `.ai-cli/config.yaml`:

```yaml
tools:
  my_tool:
    permission_required: false  # override class default
    disabled: false             # start enabled even if DISABLED_BY_DEFAULT
    allowed: true               # true is the default; false hard-blocks the tool
```

---

## Caveats

**No intra-package relative imports between your own tool files.**
Project and global tool files are loaded with `importlib.util.spec_from_file_location`
using synthetic module names.  They are *not* part of a Python package, so
`from . import helpers` will fail.  Put shared helpers in a regular installed
package, or just inline them.

**You can import `ai_cli` freely.**
The `ai_cli` package is importable when running via `python -m ai_cli`, so
`from ai_cli.tools.base import Tool` and `from ai_cli.core.workspace import
WorkspaceError` both work.

**Name collisions are allowed but logged.**
If your project tool uses the same `NAME` as a bundled or global tool, your
version wins.  A warning is printed to the log at startup.  This is
intentional — it lets you replace a bundled tool for a specific project.

**`definition()` is validated at load time.**
If `definition()` raises or returns a malformed schema, the tool is skipped
entirely (with a warning).  Test your `definition()` before deploying.

**Files starting with `_` are never loaded.**
Use this to keep helper modules (`_shared.py`) alongside your tools without
accidentally registering their classes.
