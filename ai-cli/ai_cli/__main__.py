"""
ai-cli entry point.

Run with:  python -m ai_cli [options]

Currently implemented:
  --init [--workspace PATH]   Scaffold a .ai-cli/ project directory.
  --resume [SESSION_ID]       Resume a session: pick from list, or load by ID.
  --continue                  Continue the most recent session (or start new).
  (no flags)                  Start the interactive REPL with a fresh session.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from ai_cli.cli.display import create_display
from ai_cli.cli.repl import REPL
from ai_cli.core.config_manager import ConfigError, ConfigManager
from ai_cli.core.llm_client import LLMError, create_llm_client
from ai_cli.core.permission_manager import PermissionManager
from ai_cli.core.session_manager import Session, SessionError, SessionManager
from ai_cli.core.tool_registry import ToolRegistry
from ai_cli.core.workspace import _DOT_AI_CLI, Workspace, WorkspaceError, get_global_dir
from ai_cli.utils.logging_utils import setup_logging

if TYPE_CHECKING:
    from ai_cli.cli.display import Display

_PREVIEW_LEN = 120  # max chars shown in the "unanswered message" notice

# Sentinel stored by argparse when --resume is given with no SESSION_ID argument.
# Using an object() ensures it cannot be confused with a real session-ID string.
_RESUME_PICK: object = object()


def _truncate(text: str) -> str:
    """Return *text* truncated to _PREVIEW_LEN chars with a trailing ellipsis."""
    if len(text) <= _PREVIEW_LEN:
        return text
    return text[: _PREVIEW_LEN - 1] + "…"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai-cli",
        description="AI-powered CLI assistant.",
    )
    parser.add_argument(
        "--workspace",
        metavar="PATH",
        help="Use PATH as the starting point instead of the current directory.",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialise a new .ai-cli/ project scaffold in the workspace directory.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=_RESUME_PICK,
        metavar="SESSION_ID",
        help=(
            "Resume a previous session. "
            "Without SESSION_ID, shows a list of recent sessions to pick from. "
            "With SESSION_ID, resumes that specific session directly."
        ),
    )
    parser.add_argument(
        "--continue",
        dest="continue_",
        action="store_true",
        help="Continue the most recent session. Starts a new session if none exists.",
    )

    def _positive_int(value: str) -> int:
        try:
            n = int(value)
        except ValueError as err:
            raise argparse.ArgumentTypeError(f"{value!r} is not an integer.") from err
        if n < 1:
            raise argparse.ArgumentTypeError(f"must be a positive integer (got {n}).")
        return n

    parser.add_argument(
        "--max-tool-rounds",
        dest="max_tool_rounds",
        type=_positive_int,
        metavar="N",
        help=(
            "Maximum consecutive tool-call rounds per turn (must be >= 1). "
            "Default: from config (which itself defaults to 10). "
            "When provided, overrides 'max_tool_rounds' in config."
        ),
    )
    parser.add_argument(
        "--display",
        choices=["plain", "rich"],
        default=None,
        metavar="{plain,rich}",
        help=(
            "Display backend. Default: from config (which itself defaults to 'rich'). "
            "When provided, overrides 'display_backend' in config."
        ),
    )
    return parser.parse_args()


def _load_dotenv(start: Path) -> None:
    """Load .env from the project root if one exists, otherwise no-op."""
    root = Workspace.find_root(start)
    if root is not None:
        env_file = root / ".env"
        if env_file.is_file():
            load_dotenv(env_file)


def _pick_session(
    session_manager: SessionManager,
    display: Display,
    workspace_root: Path,
    *,
    resume_id: str | None,
    resume_list: bool,
    continue_: bool,
) -> tuple[Session, bool]:
    """
    Select or create a session based on startup flags.

    Returns ``(session, resumed)`` where *resumed* is ``True`` when an
    existing session was loaded, and ``False`` when a fresh session was created.

    Raises
    ------
    SessionError
        Propagated from any of the underlying ``SessionManager`` calls
        (``load``, ``list``, ``most_recent``, or ``new``).
    """
    if resume_id is not None:
        return session_manager.load(resume_id), True

    if resume_list:
        sessions = session_manager.list(workspace_root)
        choice = display.show_session_list(sessions)
        if choice is not None:
            return session_manager.load(choice.session_id), True
        return session_manager.new(), False

    if continue_:
        session = session_manager.most_recent(workspace_root)
        if session is not None:
            return session, True
        return session_manager.new(), False

    return session_manager.new(), False


def main() -> None:
    args = parse_args()
    start = Path(args.workspace).resolve() if args.workspace else Path.cwd()

    if args.resume is not None and args.continue_:
        print(
            "Error: --resume and --continue cannot be used together.", file=sys.stderr
        )
        sys.exit(1)

    try:
        _load_dotenv(start)
        global_dir = get_global_dir()
    except ValueError as exc:
        print("Error: invalid AI_CLI_GLOBAL_DIR environment variable.", file=sys.stderr)
        print(f"Details: {exc}", file=sys.stderr)
        print(
            "Please unset AI_CLI_GLOBAL_DIR or set it to a valid, non-empty path.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not _ensure_global_dir(global_dir):
        sys.exit(0)

    if args.init:
        _cmd_init(start)
        return

    if args.resume is _RESUME_PICK:
        _cmd_repl(
            start,
            global_dir,
            resume_list=True,
            display=args.display,
            max_tool_rounds=args.max_tool_rounds,
        )
    elif args.resume is not None:
        _cmd_repl(
            start,
            global_dir,
            resume_id=str(args.resume),
            display=args.display,
            max_tool_rounds=args.max_tool_rounds,
        )
    elif args.continue_:
        _cmd_repl(
            start,
            global_dir,
            continue_=True,
            display=args.display,
            max_tool_rounds=args.max_tool_rounds,
        )
    else:
        _cmd_repl(
            start,
            global_dir,
            display=args.display,
            max_tool_rounds=args.max_tool_rounds,
        )


def _show_resume_context(session: Session, ui: Display) -> None:
    """Display context from the resumed session so the user knows where they left off.

    * If the last message was from the **assistant**: replay it through the
      display layer so it receives full formatting (Markdown, turn border, etc.).
    * If the last message was from the **user**: show a notice that it was never
      answered along with a truncated preview, so the user can decide to resend it.
    * Any other case (empty history, tool messages, errors): show only the
      session ID line.
    """
    ui.show_status(f"Resuming session {session.session_id}.")
    try:
        messages = session.get_messages()
    except SessionError:
        return

    if not messages:
        return

    last = messages[-1]
    role = last.get("role", "")
    content = last.get("content")

    if role == "assistant" and isinstance(content, str) and content.strip():
        ui.begin_assistant_turn()
        ui.stream_text(content)
        ui.end_assistant_turn()
    elif role == "user" and isinstance(content, str) and content.strip():
        ui.show_status(
            "Note: your last message was not answered — resend it to continue:"
        )
        ui.show_status(_truncate(content))


def _cmd_repl(
    start: Path,
    global_dir: Path,
    *,
    resume_id: str | None = None,
    resume_list: bool = False,
    continue_: bool = False,
    display: str | None = None,
    max_tool_rounds: int | None = None,
) -> None:
    """Bootstrap all core objects and start the interactive REPL."""
    root = Workspace.find_root(start)
    if root is None:
        print(
            f"No .ai-cli/ project found in '{start}' or any parent directory.\n"
            "Run 'ai-cli --init' to create one.",
            file=sys.stderr,
        )
        sys.exit(1)

    cli_overrides: dict = {}
    if display is not None:
        cli_overrides["display_backend"] = display
    if max_tool_rounds is not None:
        cli_overrides["max_tool_rounds"] = max_tool_rounds
    try:
        config = ConfigManager(root, cli_overrides)
        workspace = Workspace(root, config)
        llm_client = create_llm_client(config)
    except (ConfigError, WorkspaceError, LLMError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    sessions_dir = global_dir / "sessions"
    ui = create_display(config)
    permission_manager = PermissionManager(prompt_fn=ui.show_permission_prompt)
    tool_registry = ToolRegistry(workspace, config, permission_manager)

    try:
        session_manager = SessionManager(workspace, llm_client, sessions_dir)
        session, resumed = _pick_session(
            session_manager,
            ui,
            root,
            resume_id=resume_id,
            resume_list=resume_list,
            continue_=continue_,
        )
    except SessionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Set up logging before tool loading so all subsequent activity is captured.
    setup_logging(config, session.session_dir)
    tool_registry.load()

    if resumed:
        _show_resume_context(session, ui)

    repl = REPL(session, tool_registry, llm_client, ui, workspace, config)
    repl.run()


def _ensure_global_dir(global_dir: Path) -> bool:
    """
    Check that *global_dir* exists and is a directory.

    - If it is a directory: return True immediately.
    - If it exists but is not a directory (file or broken symlink): print an
      error and exit.
    - If it does not exist: prompt the user to create it.

    Returns True to continue startup, False to abort cleanly.
    """
    if global_dir.is_dir():
        return True

    if global_dir.exists() or global_dir.is_symlink():
        print(
            f"Error: global config path exists but is not a directory: {global_dir}",
            file=sys.stderr,
        )
        print(
            "Please remove or rename this path, or set AI_CLI_GLOBAL_DIR to a "
            "different directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Global config directory not found: {global_dir}\n"
        "ai-cli stores your personal settings (model config, global tools, ignore rules)\n"
        "in this directory.\n"
        "\n"
        "Tip: set the AI_CLI_GLOBAL_DIR environment variable to use a different location.\n"
    )
    try:
        answer = input("Create it now? [Y/n] ").strip().lower()
    except EOFError:
        answer = ""  # non-interactive: default to yes

    if answer not in ("", "y", "yes"):
        print("Aborted. Set AI_CLI_GLOBAL_DIR or create the directory manually.")
        return False

    try:
        Workspace.initialise_global(global_dir)
    except (WorkspaceError, OSError) as exc:
        print(f"Error creating global config directory: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Created global config directory: {global_dir}")
    print("Edit the config.yaml there to configure your default backend and model.")
    return True


def _cmd_init(path: Path) -> None:
    dot = path / _DOT_AI_CLI
    if dot.exists():
        try:
            answer = (
                input(f"'{dot}' already exists. Add any missing scaffold files? [Y/n] ")
                .strip()
                .lower()
            )
        except EOFError:
            answer = ""  # non-interactive: default to yes (proceed)
        if answer not in ("", "y", "yes"):
            print("Aborted.")
            return

    try:
        Workspace.initialise(path)
    except (WorkspaceError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Initialised ai-cli project in '{dot}'.")
    print("Edit '.ai-cli/config.yaml' to configure your backend and model.")


if __name__ == "__main__":
    main()
