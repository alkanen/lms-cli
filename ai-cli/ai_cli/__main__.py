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
import logging
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from ai_cli.cli.display import create_display
from ai_cli.cli.repl import REPL
from ai_cli.core.agent_registry import AgentRegistry, load_agent_specs
from ai_cli.core.config_manager import ConfigError, ConfigManager
from ai_cli.core.llm_client import LLMClient, LLMError, create_llm_client
from ai_cli.core.mcp_manager import MCPManager
from ai_cli.core.permission_manager import PermissionManager
from ai_cli.core.session_manager import Session, SessionError, SessionManager
from ai_cli.core.task_manager import TaskManager
from ai_cli.core.tool_registry import ToolRegistry
from ai_cli.core.workspace import _DOT_AI_CLI, Workspace, WorkspaceError, get_global_dir
from ai_cli.utils.logging_utils import setup_logging

if TYPE_CHECKING:
    from ai_cli.cli.display import Display

logger = logging.getLogger(__name__)

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
    parser.add_argument(
        "--summarize",
        metavar="FILE",
        help=(
            "Summarize FILE using the configured LLM and print the result. "
            "Uses document_embedding.summary_max_tokens from config (default 400). "
            "Useful for testing the summary document-embedding strategy."
        ),
    )
    try:
        _version = version("ai-cli")
    except PackageNotFoundError:
        _version = "unknown"
    parser.add_argument(
        "--version",
        action="version",
        version=f"ai-cli {_version}",
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

    if args.summarize is not None:
        _cmd_summarize(Path(args.summarize), start)
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


def _init_embedding_index(
    workspace: Workspace,
    config: ConfigManager,
    llm_client: LLMClient,
) -> None:
    """Build and attach an EmbeddingIndex to *workspace* if embeddings are enabled.

    Does nothing (silently) when ``embeddings.enabled`` is false or absent.
    Prints a warning and returns when required embedding dependencies cannot
    be imported (``numpy`` is required; ``xxhash`` is optional — falls back to
    ``hashlib`` when absent), when ``numpy`` is absent at
    ``SQLiteVectorStore`` construction time, or when any other construction
    error occurs.
    """
    try:
        emb_cfg = config.get_embedding_config()
    except ConfigError as exc:
        print(f"Warning: embedding config error — {exc}", file=sys.stderr)
        return
    if emb_cfg is None:
        return

    try:
        from ai_cli.core.embedding_index import EmbeddingIndex
        from ai_cli.core.embedding_provider import OpenAIEmbeddingProvider
        from ai_cli.core.vector_store import SQLiteVectorStore
    except ImportError as exc:
        print(
            f"Warning: embedding dependencies not available ({exc}). "
            "Run: pip install ai-cli[embeddings]",
            file=sys.stderr,
        )
        return

    db_path = workspace.ai_cli_dir / "embeddings" / "index.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        store = SQLiteVectorStore(db_path)
    except ImportError as exc:
        print(
            f"Warning: embedding dependencies not available ({exc}). "
            "Run: pip install ai-cli[embeddings]",
            file=sys.stderr,
        )
        return

    batch_size_raw = emb_cfg.get("batch_size")
    timeout_raw = emb_cfg.get("request_timeout")
    try:
        batch_size = (
            int(batch_size_raw)
            if batch_size_raw is not None and batch_size_raw != ""
            else 32
        )
        request_timeout = (
            float(timeout_raw)
            if timeout_raw is not None and timeout_raw != ""
            else 120.0
        )
    except (TypeError, ValueError) as exc:
        print(
            f"Warning: invalid embedding configuration value — {exc}",
            file=sys.stderr,
        )
        return

    try:
        provider = OpenAIEmbeddingProvider(
            model=emb_cfg["model"],
            base_url=emb_cfg.get("base_url"),
            api_key=emb_cfg.get("api_key"),
            batch_size=batch_size,
            request_timeout=request_timeout,
        )
        workspace.embedding_index = EmbeddingIndex(
            db_path=db_path,
            provider=provider,
            store=store,
            config=emb_cfg,
            workspace=workspace,
            llm_client=llm_client,
        )
    except Exception as exc:
        store.close()
        print(f"Warning: failed to initialise embedding index — {exc}", file=sys.stderr)


_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def _is_placeholder_only(text: str) -> bool:
    """Return True if *text* consists only of HTML comments and whitespace."""
    return not _HTML_COMMENT_RE.sub("", text).strip()


def load_system_prompt(workspace_root: Path, global_dir: Path) -> str:
    """Resolve the system prompt using a three-level lookup.

    Checked in order:

    1. ``<workspace_root>/.ai-cli/system_prompt.md`` — project-level override.
    2. ``<workspace_root>/AGENTS.md`` — industry-standard convention.
    3. ``<global_dir>/system_prompt.md`` — user-level default.

    Each candidate is skipped if the file only contains HTML comments and
    whitespace (placeholder-only), cannot be read, or contains non-UTF-8 bytes.

    Returns an empty string when none of the candidates yield usable content,
    which causes the system message to be omitted from the request entirely.
    """
    candidates: list[Path] = [
        workspace_root / _DOT_AI_CLI / "system_prompt.md",
        workspace_root / "AGENTS.md",
        global_dir / "system_prompt.md",
    ]
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if _is_placeholder_only(text):
            continue
        stripped = text.strip()
        if stripped:
            return stripped
    return ""


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
        _init_embedding_index(workspace, config, llm_client)
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

    # Resolve and apply the system prompt before the first LLM call.
    system_prompt = load_system_prompt(root, global_dir)
    if system_prompt:
        session.set_system_message(system_prompt)
        logger.debug("System prompt loaded (%d chars).", len(system_prompt))
    else:
        logger.debug("No system prompt found; omitting system message.")

    # Set up logging before tool loading so all subsequent activity is captured.
    setup_logging(config, session.session_dir)
    tool_registry.load()

    # Wire up task tools before agents so the startup validation in
    # _wire_agents can see the tasks_* tools when checking agent specs.
    # Tasks are project-scoped: they live under the project's .ai-cli/
    # directory so they survive across sessions and are shared between
    # every session opened in the same project.
    task_manager = TaskManager(workspace.ai_cli_dir)
    _wire_tasks(task_manager, tool_registry, workspace, permission_manager)

    # Wire up call_agent tool if any agent specs are configured.
    agent_registry = AgentRegistry(load_agent_specs(config), parent_display=ui)
    _wire_agents(
        agent_registry, tool_registry, workspace, permission_manager, config, llm_client
    )

    # Wire up MCP servers (connect, discover tools, register proxies).
    mcp_manager = _wire_mcp(
        global_dir, root, tool_registry, workspace, permission_manager
    )

    if resumed:
        _show_resume_context(session, ui)

    repl = REPL(
        session,
        tool_registry,
        llm_client,
        ui,
        workspace,
        config,
        agent_registry=agent_registry,
        task_manager=task_manager,
        mcp_manager=mcp_manager,
    )
    try:
        repl.run()
    finally:
        if mcp_manager is not None:
            mcp_manager.close_all()


def _wire_agents(
    agent_registry: AgentRegistry,
    tool_registry: ToolRegistry,
    workspace: Workspace,
    permission_manager: PermissionManager,
    config: ConfigManager,
    llm_client: LLMClient,
) -> None:
    """Register agent tools against *tool_registry* and validate tool references.

    Registers ``call_agent`` when at least one agent spec is configured.
    Registers ``call_agents_parallel`` only when
    ``agent_settings.allow_parallel: true`` is set in config.
    Warns (but does not abort) when an agent spec references a tool that is
    not present in the global registry.
    """
    if not agent_registry.has_agents:
        return

    from ai_cli.tools.call_agent import CallAgentTool

    tool_registry.register_instance(
        CallAgentTool(
            workspace,
            permission_manager,
            agent_registry,
            config,
            llm_client,
            tool_registry,
        )
    )

    agent_settings = config.get("agent_settings") or {}
    if agent_settings and not isinstance(agent_settings, dict):
        logger.warning(
            "Ignoring agent_settings: expected a mapping, got %s.",
            type(agent_settings).__name__,
        )
        agent_settings = {}
    if isinstance(agent_settings, dict):
        allow_parallel = agent_settings.get("allow_parallel")
        if allow_parallel is True:
            from ai_cli.tools.call_agent import CallAgentsParallelTool

            tool_registry.register_instance(
                CallAgentsParallelTool(
                    workspace,
                    permission_manager,
                    agent_registry,
                    config,
                    llm_client,
                    tool_registry,
                )
            )
        elif "allow_parallel" in agent_settings and not isinstance(
            allow_parallel, bool
        ):
            logger.warning(
                "Ignoring non-boolean agent_settings.allow_parallel=%r; expected true or false.",
                allow_parallel,
            )

    for agent_name, spec in agent_registry.specs.items():
        for tool_name in spec.tools:
            if tool_name == "call_agent":
                continue  # always excluded from sub-agents
            if tool_registry.get(tool_name) is None:
                logger.warning(
                    "Agent '%s': tool '%s' is not registered in the global registry.",
                    agent_name,
                    tool_name,
                )


def _wire_tasks(
    task_manager: TaskManager,
    tool_registry: ToolRegistry,
    workspace: Workspace,
    permission_manager: PermissionManager,
) -> None:
    """Register all six task tools against *tool_registry*.

    Task tools are always registered (no config gate).  Individual tools can
    be disabled or disallowed via the standard per-tool mechanism.
    """
    from ai_cli.tools.tasks import (
        TasksAddNoteTool,
        TasksCreateTool,
        TasksGetTool,
        TasksListTool,
        TasksMarkDoneTool,
        TasksUpdateTool,
    )

    for tool_cls in (
        TasksListTool,
        TasksGetTool,
        TasksCreateTool,
        TasksUpdateTool,
        TasksAddNoteTool,
        TasksMarkDoneTool,
    ):
        tool_registry.register_instance(
            tool_cls(task_manager, workspace, permission_manager)
        )


def _wire_mcp(
    global_dir: Path,
    project_root: Path,
    tool_registry: ToolRegistry,
    workspace: Workspace,
    permission_manager: PermissionManager,
) -> MCPManager | None:
    """Connect to all configured MCP servers and register their tools.

    Returns the :class:`MCPManager` instance (even if no servers are configured
    or all fail to connect), or ``None`` only when both config files are absent.
    Errors are logged as warnings; the CLI continues regardless.
    """
    global_mcp = global_dir / "mcp.yaml"
    project_mcp = project_root / _DOT_AI_CLI / "mcp.yaml"

    if not global_mcp.is_file() and not project_mcp.is_file():
        return None

    manager = MCPManager(
        global_config_path=global_mcp,
        project_config_path=project_mcp,  # always pass so --persist can create it
        tool_registry=tool_registry,
        workspace=workspace,
        permission_manager=permission_manager,
    )
    manager.connect_all()
    return manager


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


def _cmd_summarize(file_path: Path, start: Path) -> None:
    """Read *file_path*, call the LLM to summarise it, and print the result."""
    # Ensure warnings from _summarize_document are visible on stderr.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if not file_path.is_file():
        print(f"Error: not a file: {file_path}", file=sys.stderr)
        sys.exit(1)

    # Load config from workspace if one exists; fall back to global-only config.
    root = Workspace.find_root(start)
    try:
        if root is not None:
            config = ConfigManager(root, {})
        else:
            config = ConfigManager(None, {})
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        llm_client = create_llm_client(config)
    except (ConfigError, LLMError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"Error reading file: {exc}", file=sys.stderr)
        sys.exit(1)

    from ai_cli.core.embedding_index import _summarize_document

    emb_cfg: dict = {}
    try:
        cfg = config.get_embedding_config()
        if cfg is not None:
            emb_cfg = cfg
    except ConfigError:
        pass

    _raw_mt = (emb_cfg.get("document_embedding") or {}).get("summary_max_tokens")
    try:
        max_tokens = int(_raw_mt) if _raw_mt is not None else 400
    except (TypeError, ValueError):
        max_tokens = 400
    max_tokens = max(1, max_tokens)
    char_budget = max_tokens * 4
    truncated = len(text) > char_budget

    print(f"File:          {file_path}")
    print(f"Size:          {len(text):,} chars")
    if truncated:
        print(f"Truncated to:  {char_budget:,} chars (summary_max_tokens={max_tokens})")
    print(f"Model:          {config.get_model_config().get('model', '(unknown)')}")
    print()

    summary = _summarize_document(text, file_path, emb_cfg, llm_client)

    if summary is None:
        print("Error: summarization failed — see warnings above.", file=sys.stderr)
        sys.exit(1)

    print("--- Summary ---")
    print(summary)


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
