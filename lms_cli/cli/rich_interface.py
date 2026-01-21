"""
Rich-based CLI interface for lms-cli.

Provides a visually enhanced terminal UI using the Rich library with:
- Styled message panels for user/assistant/tool interactions
- Token usage progress bar
- Syntax-highlighted code previews
- Streaming response display
- Interactive permission prompts
"""

from datetime import datetime
import json
import re
from typing import Dict, List, Optional, Tuple

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from lms_cli.core.context import CLIContext
from lms_cli.core.file_reference_parser import FileReferenceParser
from lms_cli.core.session_handler import SessionHandler
from lms_cli.core.tool_registry import (
    TOOL_PERMISSION_YES,
    TOOL_PERMISSION_ALWAYS,
    TOOL_PERMISSION_NO,
    TOOL_PERMISSION_USER_SUGGESTION,
)


# Console instance for all output
console = Console()


class RichUI:
    """Rich UI component renderer."""

    def __init__(self, session_id: str, model: str, max_tokens: int):
        self.session_id = session_id
        self.model = model
        self.max_tokens = max_tokens
        self.tokens_used = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def render_status_bar(self) -> Panel:
        """Render the top status bar with session info and token usage."""
        table = Table.grid(padding=(0, 2), expand=True)
        table.add_column(style="bold cyan", ratio=1)
        table.add_column(ratio=2)
        table.add_column(ratio=2)

        # Token usage bar
        if self.max_tokens > 0:
            percentage = self.tokens_used / self.max_tokens
            bar_width = 30
            filled = int(percentage * bar_width)
            bar = "[green]" + "\u2588" * filled + "[/green][dim]" + "\u2591" * (bar_width - filled) + "[/dim]"
            token_display = f"Tokens: {self.tokens_used:,} / {self.max_tokens:,} {bar} {percentage:.0%}"
        else:
            token_display = f"Tokens: {self.tokens_used:,}"

        # Truncate session ID for display
        display_session = self.session_id
        if len(display_session) > 40:
            display_session = "..." + display_session[-37:]

        table.add_row(
            "[bold cyan]lms-cli[/bold cyan]",
            f"[dim]Session:[/dim] {display_session}",
            f"[dim]Model:[/dim] {self.model}",
        )
        table.add_row(
            "",
            token_display,
            f"[dim]Prompt:[/dim] {self.prompt_tokens:,} [dim]Completion:[/dim] {self.completion_tokens:,}",
        )

        return Panel(table, style="dim", padding=(0, 1))

    def update_token_usage(self, usage: Dict):
        """Update token counters from usage dict."""
        self.prompt_tokens = usage.get("prompt_tokens", 0)
        self.completion_tokens = usage.get("completion_tokens", 0)
        self.tokens_used = usage.get("total_tokens", 0)

    @staticmethod
    def render_user_message(content: str, attachments: Optional[List[str]] = None) -> Panel:
        """Render a user message panel."""
        body_parts = [content]

        if attachments:
            body_parts.append("")
            for attachment in attachments:
                body_parts.append(f"[cyan]\U0001F4CE Attached: {attachment}[/cyan]")

        return Panel(
            "\n".join(body_parts),
            title="[bold blue]You[/bold blue]",
            title_align="left",
            border_style="blue",
            padding=(0, 1),
        )

    @staticmethod
    def render_assistant_message(content: str, streaming: bool = False) -> Panel:
        """Render an assistant message panel."""
        if streaming:
            if content:
                display = Text(content)
                display.append(" \u25cf \u25cf \u25cf", style="dim")
            else:
                display = Text("\u25cf \u25cf \u25cf thinking...", style="dim")
        else:
            # Use Markdown for final display
            display = Markdown(content) if content else Text("[dim]No response[/dim]")

        return Panel(
            display,
            title="[bold green]Assistant[/bold green]",
            title_align="left",
            border_style="green",
            padding=(0, 1),
        )

    @staticmethod
    def render_tool_request(
        tool_name: str,
        params: Dict,
        preview_content: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> Panel:
        """Render a tool request panel with parameters and optional preview."""
        elements = []

        # Tool name header
        elements.append(Text(f"\u2699\ufe0f  {tool_name}", style="bold yellow"))
        elements.append(Text(""))

        # Parameters table
        params_table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        params_table.add_column("key", style="cyan", width=12)
        params_table.add_column("value", overflow="fold")

        for key, value in params.items():
            if key == "content":
                # Show truncated content
                content_preview = str(value)[:100]
                if len(str(value)) > 100:
                    content_preview += "..."
                params_table.add_row(f"{key}:", f"[dim]{content_preview}[/dim]")
            else:
                params_table.add_row(f"{key}:", str(value))

        elements.append(Panel(params_table, title="[dim]Parameters[/dim]", border_style="dim"))

        # Add syntax-highlighted preview for file operations
        if preview_content:
            preview_lines = preview_content.split("\n")
            truncated = len(preview_lines) > 12
            if truncated:
                preview_lines = preview_lines[:12]

            preview_text = "\n".join(preview_lines)
            if truncated:
                preview_text += f"\n[dim]\u22ee ... ({len(preview_lines) - 12} more lines)[/dim]"

            # Detect language from file extension
            lexer = "text"
            if file_path:
                if file_path.endswith(".py"):
                    lexer = "python"
                elif file_path.endswith((".js", ".jsx")):
                    lexer = "javascript"
                elif file_path.endswith((".ts", ".tsx")):
                    lexer = "typescript"
                elif file_path.endswith(".json"):
                    lexer = "json"
                elif file_path.endswith((".yaml", ".yml")):
                    lexer = "yaml"
                elif file_path.endswith(".md"):
                    lexer = "markdown"
                elif file_path.endswith((".sh", ".bash")):
                    lexer = "bash"
                elif file_path.endswith((".html", ".htm")):
                    lexer = "html"
                elif file_path.endswith(".css"):
                    lexer = "css"

            if lexer != "text" and not truncated:
                syntax = Syntax(
                    preview_text,
                    lexer,
                    line_numbers=True,
                    theme="monokai",
                    word_wrap=True,
                )
                elements.append(Panel(syntax, title="[dim]Preview[/dim]", border_style="dim"))
            else:
                elements.append(
                    Panel(
                        Text(preview_text),
                        title="[dim]Preview[/dim]",
                        border_style="dim",
                    )
                )

        return Panel(
            Group(*elements),
            title="[bold yellow]Tool Request[/bold yellow]",
            title_align="left",
            border_style="yellow",
            padding=(0, 1),
        )

    @staticmethod
    def render_tool_result(tool_name: str, success: bool, message: str) -> Panel:
        """Render a tool result panel."""
        if success:
            icon = "\u2713"
            style = "green"
        else:
            icon = "\u2717"
            style = "red"

        # Truncate long messages
        if len(message) > 500:
            display_message = message[:500] + "..."
        else:
            display_message = message

        return Panel(
            f"[{style}]{icon}[/{style}] [bold]{tool_name}[/bold]\n{display_message}",
            title=f"[{style}]Tool Result[/{style}]",
            title_align="left",
            border_style=style,
            padding=(0, 1),
        )

    @staticmethod
    def render_error(title: str, message: str, suggestion: Optional[str] = None) -> Panel:
        """Render an error panel."""
        body = f"[red]{message}[/red]"
        if suggestion:
            body += f"\n\n[dim]Suggestion: {suggestion}[/dim]"

        return Panel(
            body,
            title=f"[bold red]\u26a0 {title}[/bold red]",
            border_style="red",
            padding=(0, 1),
        )

    @staticmethod
    def render_session_picker(sessions: List[str]) -> Panel:
        """Render a session picker table."""
        table = Table(title="Recent Sessions", expand=True)
        table.add_column("#", style="cyan", width=4)
        table.add_column("Session ID", overflow="fold")
        table.add_column("Date/Time", width=20)

        for i, session in enumerate(sessions, 1):
            # Parse timestamp from session ID
            try:
                timestamp_str = session.rsplit("_", 1)[1]
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%Hh%Mm%Ss")
                display_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            except (IndexError, ValueError):
                display_time = "Unknown"

            # Truncate session ID for display
            display_id = session
            if len(display_id) > 50:
                display_id = display_id[:47] + "..."

            table.add_row(str(i), display_id, display_time)

        return Panel(table, border_style="cyan")

    @staticmethod
    def render_compaction_progress() -> Panel:
        """Render a compaction in progress panel."""
        return Panel(
            "[yellow]\u23f3[/yellow] Compacting conversation history...",
            title="[yellow]Compaction[/yellow]",
            border_style="yellow",
        )

    @staticmethod
    def render_help() -> Panel:
        """Render help information."""
        help_text = """[bold]Commands:[/bold]
  [cyan]@file.py[/cyan]        Include file contents in message
  [cyan]@file.py:10-20[/cyan]  Include specific lines from file
  [cyan]/compact[/cyan]        Summarize conversation history
  [cyan]/help[/cyan]           Show this help message
  [cyan]exit[/cyan]            Exit the shell

[bold]Permission Options:[/bold]
  [green]y[/green] - Yes, allow this execution
  [green]a[/green] - Always allow this tool
  [red]n[/red] - No, deny the request
  [yellow]f[/yellow] - Show full content
  [yellow]s[/yellow] - Suggest alternative"""

        return Panel(help_text, title="[bold]Help[/bold]", border_style="cyan")


def permission_request_rich(
    question: str, options: List[str]
) -> Tuple[int, str]:
    """
    Rich-based permission request dialog.

    Returns:
        Tuple of (permission_code, suggestion_string)
    """
    console.print()

    # Show the question
    console.print(f"[yellow]{question}[/yellow]")
    console.print()

    # Build options display
    options_text = [
        "[green]y[/green] Yes",
        "[green]a[/green] Always allow",
        "[red]n[/red] No",
    ]

    # Add custom options
    for i, option in enumerate(options):
        options_text.append(f"[cyan]{i + 1}[/cyan] {option}")

    options_text.append("[yellow]s[/yellow] Suggest alternative")
    options_text.append("[blue]f[/blue] Show full content")

    console.print("  " + "  |  ".join(options_text[:3]))
    if len(options_text) > 4:
        console.print("  " + "  |  ".join(options_text[3:]))

    console.print()

    # Get input
    valid_choices = ["y", "a", "n", "s", "f"] + [str(i + 1) for i in range(len(options))]
    choice = Prompt.ask(
        "[bold]Choice[/bold]",
        choices=valid_choices,
        default="y",
        show_choices=False,
    )

    permission_string = ""

    if choice == "y":
        return TOOL_PERMISSION_YES, permission_string
    elif choice == "a":
        return TOOL_PERMISSION_ALWAYS, permission_string
    elif choice == "n":
        return TOOL_PERMISSION_NO, permission_string
    elif choice == "s":
        permission_string = Prompt.ask("[bold]Enter suggested behaviour[/bold]")
        return TOOL_PERMISSION_USER_SUGGESTION, permission_string
    elif choice == "f":
        # Return special value to indicate "show full"
        return -1, "show_full"
    else:
        # Custom option selected
        option_index = int(choice) - 1
        return option_index, permission_string


def run_rich_shell(
    config: str,
    workspace_root: str,
    prompt: Optional[str],
    resume: bool,
):
    """
    Run the Rich-based interactive shell.

    Args:
        config: Path to configuration file
        workspace_root: Path to workspace directory
        prompt: Optional immediate prompt to send
        resume: Whether to resume a previous session
    """

    # Create permission callback that uses Rich UI
    def permission_callback(question: str, options: List[str]) -> Tuple[int, str]:
        return permission_request_rich(question, options)

    # Initialize context
    context = CLIContext(
        config_path=config,
        workspace_root=workspace_root,
        permission_callback=permission_callback,
    )
    context.tool_registry.load_tools()

    max_tokens = context.config["lm_studio"].get("max_tokens", 4096)
    stream = context.config["lm_studio"].get("stream", True)
    model = context.config["lm_studio"].get("model", "unknown")
    sh = None
    messages = []

    # Handle session resumption
    if resume:
        sessions = SessionHandler.list_available_sessions()
        if sessions:
            console.print()
            console.print(RichUI.render_session_picker(sessions))

            choice = Prompt.ask(
                "[bold]Select session number[/bold]",
                choices=[str(i) for i in range(1, len(sessions) + 1)] + ["c"],
                default="1",
            )

            if choice != "c":
                session_index = int(choice) - 1
                if sh := SessionHandler.restore_session(sessions[session_index]):
                    messages = sh.load_recent_history()
                    console.print(f"[green]\u2713 Restored session with {len(messages)} messages[/green]")

    # Create new session if not resuming or resume failed
    if not messages:
        sh = SessionHandler(context.workspace)
        messages = [
            {"role": "system", "content": context.lm_studio_client.system_message},
        ]
        sh.save_message(messages[-1])

    # Update context with session handler
    context.session_handler = sh

    # Initialize UI
    ui = RichUI(
        session_id=sh.session_id,
        model=model,
        max_tokens=max_tokens,
    )

    # Display startup
    console.print()
    console.print(ui.render_status_bar())
    console.print()
    console.print(
        "[dim]Type your message and press Enter. "
        "Use @file.py to include files, /help for commands, 'exit' to quit.[/dim]"
    )

    # Main loop
    while True:
        # Get available tool definitions
        tools = [
            context.tool_registry.get_tool_definition(name)
            for name in context.tool_registry.tools
        ]

        try:
            is_compaction_request = False
            attachments = []
            user_input = ""

            # Handle immediate prompt or get user input
            if prompt:
                if prompt.startswith("@"):
                    file_path = prompt[1:]
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            content = f.read()
                    except (OSError, UnicodeError) as e:
                        console.print(
                            RichUI.render_error(
                                "File error",
                                f"Could not read file '{file_path}': {e}",
                                suggestion=(
                                    "Check that the file exists, is readable and is "
                                    "valid UTF-8."
                                ),
                            )
                        )
                        prompt = None  # Clear after first use
                        continue

                    attachments.append(file_path)
                else:
                    content = prompt
                prompt = None  # Clear after first use
            else:
                console.print()
                user_input = Prompt.ask("[bold blue]>[/bold blue]").strip()

                if not user_input:
                    continue

                if user_input.lower() == "exit":
                    console.print("[dim]Goodbye![/dim]")
                    break

                if user_input == "/help":
                    console.print(RichUI.render_help())
                    continue

                if user_input.startswith("/compact"):
                    is_compaction_request = True
                    compaction_instruction = user_input[8:].strip()
                    if not compaction_instruction:
                        compaction_instruction = (
                            "Include key details, decisions made, unresolved "
                            "questions, and any important context or goals discussed. "
                            "Ensure the summary retains enough detail to allow the "
                            "conversation to continue seamlessly without losing "
                            "critical information."
                        )

                    content = (
                        "Summarize the entire message history up to this point in a "
                        "structured, concise manner. "
                        f"Instructions: {compaction_instruction}"
                    )
                else:
                    # Parse file references
                    original_input = user_input
                    content = FileReferenceParser.parse_message(user_input)

                    # Extract attachment info for display
                    file_refs = re.findall(r"@([\w./\-]+(?::\d+-\d+)?)", original_input)
                    attachments = file_refs

            # Display user message
            console.print(RichUI.render_user_message(
                user_input if not is_compaction_request else "/compact",
                attachments if attachments else None
            ))

            # Add to messages
            messages.append({"role": "user", "content": content})
            context.session_handler.save_message(messages[-1])

            # Show compaction progress if needed
            if is_compaction_request:
                console.print(RichUI.render_compaction_progress())

            # Stream the response
            response_content = ""
            first_chunk = True

            def on_chunk(chunk: str):
                nonlocal response_content, first_chunk
                response_content += chunk
                first_chunk = False

            def on_usage(usage: Dict):
                ui.update_token_usage(usage)

            # Use Live display for streaming
            with Live(
                RichUI.render_assistant_message("", streaming=True),
                console=console,
                refresh_per_second=10,
                transient=True,
            ) as live:
                def live_chunk(chunk: str):
                    nonlocal response_content
                    response_content += chunk
                    live.update(RichUI.render_assistant_message(response_content, streaming=True))

                result = context.lm_studio_client.chat_completion(
                    messages,
                    tools=tools if not is_compaction_request else None,
                    stream=stream,
                    on_chunk_callback=live_chunk if stream else on_chunk,
                    usage_callback=on_usage,
                )

            # Display final assistant message
            if result["content"]:
                console.print(RichUI.render_assistant_message(result["content"], streaming=False))

            # Update status bar with new token count
            console.print(ui.render_status_bar())

            # Add assistant message to history
            messages.append({
                "role": "assistant",
                "content": result["content"],
                **({"tool_calls": result["tool_calls"]} if result["tool_calls"] else {}),
            })
            context.session_handler.save_message(messages[-1])

            # Handle compaction
            if is_compaction_request:
                messages = context.session_handler.compact_recent_history()
                console.print("[green]\u2713 History compacted[/green]")
                continue

            # Process tool calls
            tool_calls = result["tool_calls"]

            while tool_calls:
                tool_responses = []

                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    tool_args_str = tc["function"]["arguments"]

                    try:
                        tool_args = json.loads(tool_args_str) if tool_args_str else {}
                    except json.JSONDecodeError:
                        tool_args = {"raw": tool_args_str}

                    # Get preview content for file operations
                    preview_content = None
                    file_path = None
                    if tool_name == "write_file" and "content" in tool_args:
                        preview_content = tool_args.get("content", "")
                        file_path = tool_args.get("file_path", "")
                    elif tool_name == "read_file":
                        file_path = tool_args.get("file_path", "")

                    # Display tool request
                    console.print(RichUI.render_tool_request(
                        tool_name,
                        tool_args,
                        preview_content=preview_content,
                        file_path=file_path,
                    ))

                    try:
                        # Execute tool
                        tool_result = context.tool_registry.execute_tool(
                            tool_name, tool_args_str
                        )

                        # Display success
                        result_str = json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
                        console.print(RichUI.render_tool_result(
                            tool_name,
                            success=True,
                            message=result_str[:200] + "..." if len(result_str) > 200 else result_str,
                        ))

                        tool_responses.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": json.dumps(tool_result),
                        })

                    except Exception as e:
                        error_msg = str(e)
                        console.print(RichUI.render_tool_result(
                            tool_name,
                            success=False,
                            message=error_msg,
                        ))

                        tool_responses.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": f"Error: {error_msg}",
                        })

                    # Save tool response
                    context.session_handler.save_message(tool_responses[-1])

                # Add tool responses to messages
                messages.extend(tool_responses)

                # Continue conversation with tool results
                response_content = ""

                with Live(
                    RichUI.render_assistant_message("", streaming=True),
                    console=console,
                    refresh_per_second=10,
                    transient=True,
                ) as live:
                    def live_chunk_continued(chunk: str):
                        nonlocal response_content
                        response_content += chunk
                        live.update(RichUI.render_assistant_message(response_content, streaming=True))

                    result = context.lm_studio_client.chat_completion(
                        messages,
                        tools=tools,
                        stream=stream,
                        on_chunk_callback=live_chunk_continued if stream else on_chunk,
                        usage_callback=on_usage,
                    )

                # Display final response
                if result["content"]:
                    console.print(RichUI.render_assistant_message(result["content"], streaming=False))

                # Update status bar
                console.print(ui.render_status_bar())

                # Add to messages
                messages.append({
                    "role": "assistant",
                    "content": result["content"],
                    **({"tool_calls": result["tool_calls"]} if result["tool_calls"] else {}),
                })
                context.session_handler.save_message(messages[-1])

                # Check for more tool calls
                tool_calls = result["tool_calls"]

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Type 'exit' to quit.[/dim]")
            continue
        except Exception as e:
            console.print(RichUI.render_error(
                "Error",
                str(e),
                suggestion="Check your configuration and try again.",
            ))
