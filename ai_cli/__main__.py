"""
ai-cli entry point.

Run with:  python -m ai_cli [options]

Currently implemented:
  --init [--workspace PATH]   Scaffold a .ai-cli/ project directory.
  --ask [PROMPT]              One-shot: print the model's answer and exit.

Settings can be pointed anywhere, in every mode:
  --ai-cli-dir DIR            Use DIR as the project config directory.
  --env FILE                  Load environment variables from FILE.
  --system-prompt FILE        Use FILE as the system prompt.
  --no-system-prompt          Send no system prompt.
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
from ai_cli.cli.repl import REPL, skill_aliases_for_registry
from ai_cli.core.agent_registry import AgentRegistry, load_agent_specs
from ai_cli.core.config_manager import ConfigError, ConfigManager
from ai_cli.core.llm_client import (
    LLMClient,
    LLMError,
    _ThinkTagParser,
    create_llm_client,
)
from ai_cli.core.mcp_manager import MCPManager
from ai_cli.core.permission_manager import PermissionManager
from ai_cli.core.session_manager import Session, SessionError, SessionManager
from ai_cli.core.skill_registry import SkillRegistry
from ai_cli.core.task_manager import TaskManager
from ai_cli.core.tool_registry import ToolRegistry
from ai_cli.core.workspace import _DOT_AI_CLI, Workspace, WorkspaceError, get_global_dir
from ai_cli.utils.logging_utils import setup_logging
from ai_cli.utils.spinner import Spinner

if TYPE_CHECKING:
    from ai_cli.cli.display import Display

logger = logging.getLogger(__name__)

_PREVIEW_LEN = 120  # max chars shown in the "unanswered message" notice

# Sentinel stored by argparse when --resume is given with no SESSION_ID argument.
# Using an object() ensures it cannot be confused with a real session-ID string.
_RESUME_PICK: object = object()

# Sentinel stored by argparse when --ask is given with no PROMPT argument;
# the prompt is then read from stdin.
_ASK_STDIN: object = object()


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
        "--ask",
        nargs="?",
        const=_ASK_STDIN,
        metavar="PROMPT",
        help=(
            "Send PROMPT to the configured LLM, print the answer, and exit. "
            "Without PROMPT, the prompt is read from stdin. "
            "No tools, no session, no REPL — reasoning output is stripped so "
            "only the final answer reaches stdout."
        ),
    )
    parser.add_argument(
        "--ai-cli-dir",
        dest="ai_cli_dir",
        metavar="DIR",
        help=(
            "Use DIR as the project config directory, verbatim, instead of "
            "locating '<workspace>/.ai-cli/'. DIR need not be named '.ai-cli' "
            "and no parent directories are searched. Everything project-scoped "
            "follows it: config.yaml, system_prompt.md, tools/, skills/, "
            "mcp.yaml, .ignore and tasks.json. With --init, the scaffold is "
            "created there."
        ),
    )
    parser.add_argument(
        "--env",
        metavar="FILE",
        help=(
            "Load environment variables from FILE instead of discovering the "
            "project's .env. Values in FILE override variables already set in "
            "the environment."
        ),
    )
    _prompt_group = parser.add_mutually_exclusive_group()
    _prompt_group.add_argument(
        "--system-prompt",
        dest="system_prompt",
        metavar="FILE",
        help=(
            "Use FILE as the system prompt, overriding the usual lookup. "
            "Not applicable to --init or --summarize."
        ),
    )
    _prompt_group.add_argument(
        "--no-system-prompt",
        dest="no_system_prompt",
        action="store_true",
        help=(
            "Send no system prompt, even when a system_prompt.md would "
            "otherwise be found. Skills guidance, which describes a callable "
            "tool rather than persona instructions, is still included in the "
            "REPL. Not applicable to --init or --summarize."
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


def _load_dotenv(start: Path, env_file: Path | None = None) -> None:
    """Load environment variables from a .env file.

    With *env_file*, that file is loaded and project discovery is skipped; the
    file must exist, and its values override variables already present in the
    environment (an explicit path is an explicit intent).  Without it, the
    project root's ``.env`` is loaded when one exists — those values do *not*
    override the ambient environment — and it is a no-op when no project root
    or no ``.env`` is found.
    """
    if env_file is not None:
        if not env_file.is_file():
            print(f"Error: env file not found: {env_file}", file=sys.stderr)
            sys.exit(1)
        load_dotenv(env_file, override=True)
        return

    root = Workspace.find_root(start)
    if root is not None:
        discovered = root / ".env"
        if discovered.is_file():
            load_dotenv(discovered)


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

    # --init writes a scaffold and --summarize builds its own prompt, so
    # neither has a system prompt to override.  Rejecting is more honest than
    # accepting a flag that would silently do nothing.
    promptless = "--init" if args.init else ("--summarize" if args.summarize else "")
    if promptless:
        unusable = [
            name
            for name, given in (
                ("--system-prompt", args.system_prompt is not None),
                ("--no-system-prompt", args.no_system_prompt),
            )
            if given
        ]
        if unusable:
            print(
                f"Error: {', '.join(unusable)} cannot be used with "
                f"{promptless}, which sends no system prompt.",
                file=sys.stderr,
            )
            sys.exit(1)

    config_dir = Path(args.ai_cli_dir) if args.ai_cli_dir else None
    # --init creates the directory; every other mode must find it already there.
    if config_dir is not None and not args.init and not config_dir.is_dir():
        print(f"Error: not a directory: {config_dir}", file=sys.stderr)
        sys.exit(1)

    system_prompt_file = Path(args.system_prompt) if args.system_prompt else None

    try:
        _load_dotenv(start, Path(args.env) if args.env else None)
        global_dir = get_global_dir()
    except ValueError as exc:
        print("Error: invalid AI_CLI_GLOBAL_DIR environment variable.", file=sys.stderr)
        print(f"Details: {exc}", file=sys.stderr)
        print(
            "Please unset AI_CLI_GLOBAL_DIR or set it to a valid, non-empty path.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Dispatched before _ensure_global_dir: --ask is a non-interactive
    # one-shot, so it must never block on the "create global dir?" prompt.
    if args.ask is not None:
        prompt = sys.stdin.read() if args.ask is _ASK_STDIN else str(args.ask)
        _cmd_ask(
            prompt,
            start,
            global_dir,
            config_dir=config_dir,
            system_prompt_file=system_prompt_file,
            no_system_prompt=args.no_system_prompt,
        )
        return

    if not _ensure_global_dir(global_dir):
        sys.exit(0)

    if args.init:
        _cmd_init(start, config_dir)
        return

    if args.summarize is not None:
        _cmd_summarize(Path(args.summarize), start, config_dir)
        return

    repl_kwargs: dict = {
        "display": args.display,
        "max_tool_rounds": args.max_tool_rounds,
        "config_dir": config_dir,
        "system_prompt_file": system_prompt_file,
        "no_system_prompt": args.no_system_prompt,
    }
    if args.resume is _RESUME_PICK:
        _cmd_repl(start, global_dir, resume_list=True, **repl_kwargs)
    elif args.resume is not None:
        _cmd_repl(start, global_dir, resume_id=str(args.resume), **repl_kwargs)
    elif args.continue_:
        _cmd_repl(start, global_dir, continue_=True, **repl_kwargs)
    else:
        _cmd_repl(start, global_dir, **repl_kwargs)


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


def load_system_prompt(
    workspace_root: Path | None,
    global_dir: Path,
    config_dir: Path | None = None,
) -> str:
    """Resolve the system prompt using a three-level lookup.

    Checked in order:

    1. ``<config dir>/system_prompt.md`` — project-level override, where the
       config dir is *config_dir* when given, else
       ``<workspace_root>/.ai-cli/``.
    2. ``<workspace_root>/AGENTS.md`` — industry-standard convention.  Skipped
       when there is no workspace root, since it is a property of the project
       tree rather than of the config bundle.
    3. ``<global_dir>/system_prompt.md`` — user-level default.

    Each candidate is skipped if the file only contains HTML comments and
    whitespace (placeholder-only), cannot be read, or contains non-UTF-8 bytes.

    Returns an empty string when none of the candidates yield usable content,
    which causes the system message to be omitted from the request entirely.
    """
    candidates: list[Path] = []
    if config_dir is not None:
        candidates.append(config_dir / "system_prompt.md")
    elif workspace_root is not None:
        candidates.append(workspace_root / _DOT_AI_CLI / "system_prompt.md")
    if workspace_root is not None:
        candidates.append(workspace_root / "AGENTS.md")
    candidates.append(global_dir / "system_prompt.md")
    return _first_usable_prompt(candidates)


def _first_usable_prompt(candidates: list[Path]) -> str:
    """Return the first candidate's stripped content, or ``""`` if none is usable.

    A candidate is skipped when it cannot be read, holds non-UTF-8 bytes, is
    blank, or contains only HTML comments (a placeholder scaffold file).
    """
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


def resolve_system_prompt(
    global_dir: Path,
    *,
    explicit: Path | None,
    disabled: bool,
    config_dir: Path | None,
    workspace_root: Path | None,
) -> str:
    """Resolve the base system prompt, honouring the CLI overrides.

    Precedence: ``--no-system-prompt`` beats ``--system-prompt FILE``, which
    beats :func:`load_system_prompt`.  An explicit file is read verbatim (no
    placeholder-only filtering) and a missing or unreadable one is a hard
    error — naming a file that cannot be used is a mistake worth reporting,
    not something to silently fall back from.

    Returns ``""`` when no system message should be sent.  For the REPL that is
    the *base* prompt only: skills guidance is still appended by
    :func:`compose_system_prompt`, since it describes a tool the model can call
    rather than persona instructions the user asked to drop.
    """
    if disabled:
        return ""

    if explicit is not None:
        try:
            return explicit.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            print(f"Error: cannot read system prompt file: {exc}", file=sys.stderr)
            sys.exit(1)

    return load_system_prompt(workspace_root, global_dir, config_dir)


def compose_system_prompt(base_prompt: str, skills: SkillRegistry) -> str:
    """Return the active session system prompt with optional skills guidance."""
    base = base_prompt.strip()
    if not skills.has_skills:
        return base

    skill_rows = [
        f"- {name}: {spec.description}" for name, spec in sorted(skills.items())
    ]
    skills_section = "\n".join(
        [
            "## Skills",
            "",
            "When a task matches one of the skills below, call the `skills` tool before taking action.",
            "Use exactly one skill at a time.",
            "",
            "`skills(name)` behavior:",
            "- Input: canonical skill name (exact match).",
            "- Result envelope: top-level `status` (`success` or `error`) and `data`.",
            "- Success data fields: `data.name`, `data.description`, `data.instructions`, `data.base_dir`.",
            "- Not-found data fields: `data.found=false`, `data.requested_name`, `data.available_skills`.",
            "",
            "For any file reads driven by skill instructions, call `read_file` with the returned `data.base_dir`.",
            "`read_file` enforces that paths resolve inside `data.base_dir`.",
            "",
            "Available skills:",
            *skill_rows,
        ]
    )

    if not base:
        return skills_section
    return f"{base}\n\n{skills_section}"


def _cmd_repl(
    start: Path,
    global_dir: Path,
    *,
    resume_id: str | None = None,
    resume_list: bool = False,
    continue_: bool = False,
    display: str | None = None,
    max_tool_rounds: int | None = None,
    config_dir: Path | None = None,
    system_prompt_file: Path | None = None,
    no_system_prompt: bool = False,
) -> None:
    """Bootstrap all core objects and start the interactive REPL.

    With *config_dir*, settings are read from that directory instead of a
    discovered ``.ai-cli/``, and the workspace root becomes *start* itself — no
    project scaffold has to exist anywhere in the tree.  Everything
    project-scoped follows the config dir, because it all hangs off
    :attr:`Workspace.ai_cli_dir`.
    """
    if config_dir is not None:
        # The config bundle is explicit, so the tree needs no .ai-cli/ at all:
        # files come from *start*, settings come from *config_dir*.
        root = start
    else:
        found = Workspace.find_root(start)
        if found is None:
            print(
                f"No .ai-cli/ project found in '{start}' or any parent directory.\n"
                "Run 'ai-cli --init' to create one, or point --ai-cli-dir at an "
                "existing config directory.",
                file=sys.stderr,
            )
            sys.exit(1)
        root = found

    cli_overrides: dict = {}
    if display is not None:
        cli_overrides["display_backend"] = display
    if max_tool_rounds is not None:
        cli_overrides["max_tool_rounds"] = max_tool_rounds
    try:
        config = ConfigManager(root, cli_overrides, config_dir=config_dir)
        workspace = Workspace(root, config, ai_cli_dir=config_dir)
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

    # Load skills registry early so validation warnings are visible at startup.
    # PR1 scope: discovery + validation + warning surface.
    # PR2 scope: model-facing ``skills`` tool registration when skills exist.
    skills = SkillRegistry.load(root, global_dir=global_dir, config_dir=config_dir)
    for warning in skills.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    _, alias_warnings = skill_aliases_for_registry(skills)
    for warning in alias_warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    _wire_skills(skills, tool_registry, workspace, permission_manager)

    def _build_active_system_prompt(current_skills: SkillRegistry) -> str:
        base_prompt = resolve_system_prompt(
            global_dir,
            explicit=system_prompt_file,
            disabled=no_system_prompt,
            config_dir=config_dir,
            workspace_root=root,
        )
        return compose_system_prompt(base_prompt, current_skills)

    # Resolve and apply the active session system prompt before the first LLM call.
    system_prompt = _build_active_system_prompt(skills)
    if system_prompt:
        session.set_system_message(system_prompt)
        logger.debug("System prompt loaded (%d chars).", len(system_prompt))
    else:
        logger.debug("No system prompt found; omitting system message.")

    # Wire up call_agent tool if any agent specs are configured.
    agent_registry = AgentRegistry(load_agent_specs(config), parent_display=ui)
    _wire_agents(
        agent_registry, tool_registry, workspace, permission_manager, config, llm_client
    )

    # Wire up MCP servers (connect, discover tools, register proxies).
    mcp_manager = _wire_mcp(
        global_dir, workspace.ai_cli_dir, tool_registry, workspace, permission_manager
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
        skill_registry=skills,
        system_prompt_builder=_build_active_system_prompt,
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
    """Register all task tools against *tool_registry*.

    Task tools are always registered (no config gate).  Individual tools can
    be disabled or disallowed via the standard per-tool mechanism.
    """
    from ai_cli.tools.tasks import (
        TasksAddNoteTool,
        TasksCreateTool,
        TasksGetTool,
        TasksListTool,
        TasksMarkDoneTool,
        TasksObsoleteNoteTool,
        TasksUpdateTool,
    )

    for tool_cls in (
        TasksListTool,
        TasksGetTool,
        TasksCreateTool,
        TasksUpdateTool,
        TasksAddNoteTool,
        TasksObsoleteNoteTool,
        TasksMarkDoneTool,
    ):
        tool_registry.register_instance(
            tool_cls(task_manager, workspace, permission_manager)
        )


def _wire_skills(
    skills: SkillRegistry,
    tool_registry: ToolRegistry,
    workspace: Workspace,
    permission_manager: PermissionManager,
) -> None:
    """Register skills tool and bind skills context into read_file."""
    read_file_tool = tool_registry.get("read_file")
    if read_file_tool is not None:
        set_skill_registry = getattr(read_file_tool, "set_skill_registry", None)
        if callable(set_skill_registry):
            set_skill_registry(skills if skills.has_skills else None)

    if not skills.has_skills:
        return

    from ai_cli.tools.skills import SkillsTool

    tool_registry.register_instance(SkillsTool(skills, workspace, permission_manager))


def _wire_mcp(
    global_dir: Path,
    project_config_dir: Path,
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
    project_mcp = project_config_dir / "mcp.yaml"

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


def _cmd_ask(
    prompt: str,
    start: Path,
    global_dir: Path,
    *,
    config_dir: Path | None = None,
    system_prompt_file: Path | None = None,
    no_system_prompt: bool = False,
) -> None:
    """Send *prompt* to the configured LLM and print only the answer text.

    A one-shot mode: no session, no tools, no REPL.  ``reasoning`` chunks are
    dropped and ``<think>…</think>`` tags are stripped (even when
    ``extract_think_tags`` is off in config), so stdout carries the cleaned-up
    answer and nothing else.  Leading and trailing whitespace is trimmed while
    still streaming, so piping the output stays predictable.

    While waiting for the model a spinner is drawn on **stderr** (never stdout,
    and only when stderr is a terminal), so an interactive caller can see the
    command has not hung without corrupting captured output.

    A system prompt is resolved via :func:`_resolve_ask_system_prompt` and sent
    as a leading system message when one is found.  Skills guidance is never
    appended — this mode exposes no tools for a skill to drive.

    With *config_dir*, ``config.yaml`` and ``system_prompt.md`` are read from
    that directory verbatim and no project root is searched for, so the mode can
    run against a self-contained bundle owned by another application.
    """
    prompt = prompt.strip()
    if not prompt:
        print("Error: empty prompt.", file=sys.stderr)
        sys.exit(1)

    if config_dir is not None and not config_dir.is_dir():
        print(f"Error: not a directory: {config_dir}", file=sys.stderr)
        sys.exit(1)

    # With an explicit config dir, take it verbatim; otherwise discover the
    # workspace, falling back to global-only config when there is none.
    root = None if config_dir is not None else Workspace.find_root(start)
    try:
        config = ConfigManager(root, {}, config_dir=config_dir)
        llm_client = create_llm_client(config)
    except (ConfigError, LLMError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    system_prompt = resolve_system_prompt(
        global_dir,
        explicit=system_prompt_file,
        disabled=no_system_prompt,
        config_dir=config_dir,
        workspace_root=root,
    )
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    parser = _ThinkTagParser()
    spinner = Spinner()
    started = False  # True once non-whitespace output has been written
    pending = ""  # trailing whitespace held back until more text follows

    def emit(text: str) -> None:
        nonlocal started, pending
        if not text:
            return
        if not started:
            text = text.lstrip()
            if not text:
                return
            started = True
        stripped = text.rstrip()
        if stripped:
            # Erase the spinner before the first byte of the answer reaches the
            # terminal, so it cannot overwrite the start of the output.
            spinner.stop()
            sys.stdout.write(pending + stripped)
            sys.stdout.flush()
            pending = text[len(stripped) :]
        else:
            pending += text

    # The spinner covers the whole wait: reasoning tokens are discarded, so a
    # reasoning model can stay silent on stdout for a long time before the
    # first word of the answer appears.
    spinner.start()
    try:
        for chunk in llm_client.send(messages, []):
            if chunk.get("type") != "text":
                continue  # drop reasoning, tool-call and done chunks
            for part in parser.feed(chunk.get("delta", "")):
                if part["type"] == "text":
                    emit(part["delta"])
        for part in parser.flush():
            if part["type"] == "text":
                emit(part["delta"])
    except LLMError as exc:
        spinner.stop()  # erase before the error, not after it
        if started:
            sys.stdout.write("\n")
            sys.stdout.flush()
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        spinner.stop()

    if not started:
        print("Error: the model returned an empty response.", file=sys.stderr)
        sys.exit(1)

    sys.stdout.write("\n")
    sys.stdout.flush()


def _cmd_summarize(
    file_path: Path, start: Path, config_dir: Path | None = None
) -> None:
    """Read *file_path*, call the LLM to summarise it, and print the result.

    *config_dir* overrides where ``config.yaml`` is read from, so a summary can
    be run against a standalone config bundle.
    """
    # Ensure warnings from _summarize_document are visible on stderr.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if not file_path.is_file():
        print(f"Error: not a file: {file_path}", file=sys.stderr)
        sys.exit(1)

    # Load config from workspace if one exists; fall back to global-only config.
    root = None if config_dir is not None else Workspace.find_root(start)
    try:
        config = ConfigManager(root, {}, config_dir=config_dir)
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


def _cmd_init(path: Path, config_dir: Path | None = None) -> None:
    """Scaffold a project config directory.

    Creates ``<path>/.ai-cli/`` unless *config_dir* names somewhere else, in
    which case the scaffold is written there verbatim.
    """
    dot = config_dir if config_dir is not None else path / _DOT_AI_CLI
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
        Workspace.initialise(path, config_dir)
    except (WorkspaceError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Initialised ai-cli project in '{dot}'.")
    print(f"Edit '{dot / 'config.yaml'}' to configure your backend and model.")


if __name__ == "__main__":
    main()
