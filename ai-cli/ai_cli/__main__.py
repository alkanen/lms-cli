"""
ai-cli entry point.

Run with:  python -m ai_cli [options]

Currently implemented:
  --init [--workspace PATH]   Scaffold a .ai-cli/ project directory.
  (no flags)                  Start the interactive REPL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from ai_cli.cli.display import create_display
from ai_cli.cli.repl import REPL
from ai_cli.core.config_manager import ConfigError, ConfigManager
from ai_cli.core.llm_client import LLMError, create_llm_client
from ai_cli.core.permission_manager import PermissionManager
from ai_cli.core.session_manager import SessionError, SessionManager
from ai_cli.core.tool_registry import ToolRegistry
from ai_cli.core.workspace import _DOT_AI_CLI, Workspace, WorkspaceError, get_global_dir


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
    return parser.parse_args()


def _load_dotenv(start: Path) -> None:
    """Load .env from the project root if one exists, otherwise no-op."""
    root = Workspace.find_root(start)
    if root is not None:
        env_file = root / ".env"
        if env_file.is_file():
            load_dotenv(env_file)


def main() -> None:
    args = parse_args()
    start = Path(args.workspace).resolve() if args.workspace else Path.cwd()

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

    _cmd_repl(start, global_dir)


def _cmd_repl(start: Path, global_dir: Path) -> None:
    """Bootstrap all core objects and start the interactive REPL."""
    root = Workspace.find_root(start)
    if root is None:
        print(
            f"No .ai-cli/ project found in '{start}' or any parent directory.\n"
            "Run 'ai-cli --init' to create one.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        config = ConfigManager(root, {})
        workspace = Workspace(root, config)
        llm_client = create_llm_client(config)
    except (ConfigError, WorkspaceError, LLMError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    sessions_dir = global_dir / "sessions"
    display = create_display(config)
    permission_manager = PermissionManager(prompt_fn=display.show_permission_prompt)
    tool_registry = ToolRegistry(workspace, config, permission_manager)
    tool_registry.load()

    try:
        session_manager = SessionManager(workspace, llm_client, sessions_dir)
        session = session_manager.new()
    except SessionError as exc:
        print(f"Error creating session: {exc}", file=sys.stderr)
        sys.exit(1)

    repl = REPL(session, tool_registry, llm_client, display, workspace)
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
