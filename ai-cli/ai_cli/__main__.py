"""
ai-cli entry point.

Run with:  python -m ai_cli [options]

Currently implemented:
  --init [--workspace PATH]   Scaffold a .ai-cli/ project directory.

Everything else prints "not yet implemented" and exits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from ai_cli.core.workspace import _DOT_AI_CLI, Workspace, WorkspaceError


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

    _load_dotenv(start)

    if args.init:
        _cmd_init(start)
        return

    print("ai-cli: REPL not yet implemented.", file=sys.stderr)
    sys.exit(1)


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
