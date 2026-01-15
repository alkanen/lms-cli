from datetime import datetime
from typing import List, Tuple

import click
import enquiries
import json
import yaml

from lms_cli.core.context import CLIContext
from lms_cli.core.embedding_manager import EmbeddingManager
from lms_cli.core.file_reference_parser import FileReferenceParser
from lms_cli.core.lm_studio_client import LMStudioClient
from lms_cli.core.session_handler import SessionHandler
from lms_cli.core.tool_registry import ToolRegistry
from lms_cli.core.tool_registry import (
    TOOL_PERMISSION_YES,
    TOOL_PERMISSION_ALWAYS,
    TOOL_PERMISSION_NO,
    TOOL_PERMISSION_USER_SUGGESTION,
)
from lms_cli.core.workspace import Workspace


class Choice:
    def __init__(self, i: int, message: str):
        self.i = i
        self.message = message

    def __str__(self):
        return self.message


@click.group()
def cli():
    """Interactive CLI code assistant"""


@cli.command()
@click.option("--workspace", "workspace_root", default=".", help="Workspace directory")
@click.option(
    "--config", help="Optional configuration file", default="config/config.yaml"
)
@click.option(
    "--resume",
    "resume",
    flag_value=True,
    default=False,
    help="Resume a previous session for this workspace",
)
@click.option(
    "--restore",
    "resume",
    flag_value=True,
    default=False,
    help="Alias for '--resume'",
)
@click.option(
    "--continue",
    "resume",
    flag_value=True,
    default=False,
    help="Alias for '--resume'",
)
@click.option(
    "--prompt",
    help="Immediate prompt, either a string to send or '@file'",
    default=None,
)
def context(workspace_root: str, config: str, resume: bool, prompt: str):
    context = CLIContext(config_path=config, workspace_root=workspace_root)

    messages = [
        {
            "role": "system",
            "content": "Pretend to be a mystic oracle asking questions from gullible visitors",
        },
        {"role": "user", "content": prompt},
    ]

    response = context.lm_studio_client.chat_completion(messages, stream=False)
    click.echo(response["content"])


@cli.command()
@click.option("--workspace", "workspace_root", default=".", help="Workspace directory")
@click.option("--excluded", multiple=True, help="Folders to exclude")
@click.option(
    "--config", help="Optional configuration file", default="config/config.yaml"
)
def init(workspace_root, excluded, config):
    """Initialize the workspace and create embeddings index"""
    context = CLIContext(config_path=config, workspace_root=workspace_root)

    if not click.confirm("This will create an embedding index of your code. Continue?"):
        return

    included_set = context.embedding_manager.inclusion_paths
    excluded_set = set(excluded)
    excluded_set.update(context.embedding_manager.exclusion_paths)

    # Get all files in workspace
    files = context.workspace.list_files(
        included_folders=included_set, excluded_folders=excluded_set
    )
    print(f"Found {len(files)} files to process")

    # Process files and get embeddings
    embeddings = []
    metadata = []

    context.embedding_manager.initialize_index()

    for file_path in files:
        if not file_path.is_file():
            # Skip folders etc
            continue

        if "__pycache__" in str(file_path):
            continue

        if str(file_path).endswith("~"):
            continue

        print(f"Reading {file_path}")

        content = context.workspace.read_file(file_path)
        embedding = context.lm_studio_client.get_embedding(content, is_query=False)

        embeddings.append(embedding)
        filename = str(file_path.relative_to(context.workspace.root_path))
        metadata.append(
            {
                "file": filename,
                "content": content[:500] + "...",  # Store first 500 chars as preview
            }
        )

    # Add to index
    context.embedding_manager.add_embeddings(embeddings, metadata)
    print("Embedding index created successfully")


@cli.command()
@click.option("--query", prompt="Enter your request")
@click.option(
    "--num-files", "-n", default=3, help="Maximum number of files to embed in request"
)
@click.option(
    "--config", help="Optional configuration file", default="config/config.yaml"
)
@click.option("--workspace", "workspace_root", default=".", help="Workspace directory")
def ask(query: str, num_files: int, config: str, workspace_root: str):
    """Ask the AI about your code"""
    context = CLIContext(config_path=config, workspace_root=workspace_root)

    # First check if we have an index
    try:
        context.embedding_manager.initialize_index()

        # Get query embedding
        query_embedding = context.lm_studio_client.get_embedding(query, is_query=True)

        # Search for relevant files
        results = context.embedding_manager.search(query_embedding, k=num_files)
        print(f"\nFound {len(results)} potentially relevant files:")

        for i, result in enumerate(results, 1):
            print(
                f"{i}. {result['metadata']['file']} "
                f"(similarity: {result['similarity']:.2f})"
            )

        # Prepare context
        embedding_context = "\n\n".join(
            f"File: {result['metadata']['file']}\n"
            f"Content:\n{result['metadata']['content']}"
            for result in results
        )

    except Exception as e:
        print(f"Could not use embeddings index: {e}")
        embedding_context = "No relevant files found in the embedding index."

    # Prepare messages for chat completion
    messages = [
        {"role": "system", "content": context.lm_studio_client.system_message},
        {
            "role": "user",
            "content": f"Context:\n{embedding_context}\n\nQuestion: {query}",
        },
    ]

    # Get response with streaming
    print("\nAssistant:")

    def output_chunk(chunk: str):
        click.echo(chunk, nl=False)

    result = context.lm_studio_client.chat_completion(
        messages, stream=True, on_chunk_callback=output_chunk
    )
    print()  # Newline after streaming

    if result["content"]:
        messages.append({"role": "assistant", "content": result["content"]})


@cli.command()
@click.option(
    "--resume",
    "resume",
    flag_value=True,
    default=False,
)
@click.option(
    "--prompt",
    help="Immediate prompt, either a string to send or '@file'",
    default=None,
)
@click.option(
    "--config", help="Optional configuration file", default="config/config.yaml"
)
@click.option(
    "--workspace",
    "workspace_root",
    help="Path to workspace folder",
    default=".",
)
def shell(config: str, workspace_root: str, prompt: str, resume):
    """Interactive shell mode"""

    def permission_request(question: str, options: List[str]) -> Tuple[int, str]:
        # Default options
        choices = [
            Choice(TOOL_PERMISSION_YES, "Yes"),
            Choice(TOOL_PERMISSION_ALWAYS, "Always allow tool"),
            Choice(TOOL_PERMISSION_NO, "No"),
        ]

        for i, option in enumerate(options):
            choices.append(Choice(i, option))

        choices.append(
            Choice(TOOL_PERMISSION_USER_SUGGESTION, "Abort and suggest something else")
        )

        permission_string = "<no permisson string>"
        choice = enquiries.choose(question, choices)
        if choice.i == TOOL_PERMISSION_USER_SUGGESTION:
            # permission_string = click.prompt(" Enter suggested behaviour:", type=str)
            permission_string = input(" Enter suggested behaviour: ")

        return choice.i, permission_string

    context = CLIContext(
        config_path=config,
        workspace_root=workspace_root,
        permission_callback=permission_request
    )
    context.tool_registry.load_tools()
    max_tokens = context.config["lm_studio"].get("max_tokens", 4096)
    stream = context.config["lm_studio"].get("stream", False)
    sh = None

    messages = []

    # Prepare initial messages
    if resume:
        sessions = SessionHandler.list_available_sessions()
        session_choices = [
            Choice(
                i,
                f'{datetime.strptime(session.rsplit("_", 1)[1], "%Y-%m-%dT%Hh%Mm%Ss")}',
            )
            for i, session in enumerate(sessions)
        ]
        if sessions:
            session = enquiries.choose(
                "which session should be restored?", session_choices
            )
            if sh := SessionHandler.restore_session(sessions[session.i]):
                messages = sh.load_recent_history()

    if not messages:
        # Create new session handler and save system message if necessary
        sh = SessionHandler(context.workspace)
        messages = [
            {"role": "system", "content": context.lm_studio_client.system_message},
        ]
        sh.save_message(messages[-1])

    # Replace the default session handler to support resuming etc
    context.session_handler = sh

    print("Starting interactive shell. Type 'exit' to quit.")

    while True:
        # Get available tool definitions
        tools = [
            context.tool_registry.get_tool_definition(name)
            for name in context.tool_registry.tools
        ]

        try:
            # This is an ugly hack
            is_compaction_request = False

            if prompt:
                if prompt.startswith("@"):
                    with open(prompt[1:]) as f:
                        content = f.read()
                else:
                    content = prompt

                prompt = None
            else:
                user_input = input("\n> ").strip()
                if not user_input or user_input.lower() == "exit":
                    break

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
                    content = FileReferenceParser.parse_message(user_input)

            # Add to messages, begin with provided prompt if any
            messages.append({"role": "user", "content": content})

            # Save user message to session
            context.session_handler.save_message(messages[-1])

            # Define callback to output chunks as they arrive
            first_chunk = True

            def output_chunk(chunk: str):
                nonlocal first_chunk
                if first_chunk:
                    click.echo("\nAssistant:\n")
                click.echo(chunk, nl=False)
                first_chunk = False

            def output_usage(usage: dict):
                # Don't write any usage data if we haven't gotten a proper reply
                nonlocal first_chunk
                if first_chunk:
                    trigger = "[Tool call]"
                else:
                    trigger = "[Usage]"

                # {'prompt_tokens': 537, 'completion_tokens': 42, 'total_tokens': 579}
                prompt_tokens = usage["prompt_tokens"]
                completion_tokens = usage["completion_tokens"]
                total_tokens = usage["total_tokens"]

                click.echo(
                    f"\n{trigger} Prompt tokens: {prompt_tokens}, "
                    f"Completion tokens: {completion_tokens}, "
                    f"Total tokens: {total_tokens} "
                    f"out of {max_tokens} ({100 * total_tokens / max_tokens:.2f} %)"
                )

            # Get response with streaming and tool support
            result = context.lm_studio_client.chat_completion(
                messages,
                tools=tools if not is_compaction_request else None,
                stream=stream,
                on_chunk_callback=output_chunk,
                usage_callback=output_usage,
            )

            # Add to messages with all fields (including tool_calls if present)
            messages.append(
                {
                    "role": "assistant",
                    "content": result["content"],
                    **(
                        {"tool_calls": result["tool_calls"]}
                        if result["tool_calls"]
                        else {}
                    ),
                }
            )

            # Save assistant message to session
            context.session_handler.save_message(messages[-1])

            # Replace history if this is a compaction request
            if is_compaction_request:
                messages = context.session_handler.compact_recent_history()

            # Get tool calls from result
            tool_calls = result["tool_calls"]

            while tool_calls:
                # Execute tools and collect responses
                tool_responses = []
                for tc in tool_calls:
                    try:
                        result = context.tool_registry.execute_tool(
                            tc["function"]["name"], tc["function"]["arguments"]
                        )

                        # Add tool response to messages
                        tool_responses.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": json.dumps(result),
                            }
                        )
                    except Exception as e:
                        print(
                            f"Tool call `{tc['function']['name']}({tc['function']['arguments']}) failed: {str(e)}"
                        )
                        tool_responses.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": f"Error {str(e)}",
                            }
                        )
                        raise e

                    # Save tool responses to session
                    context.session_handler.save_message(tool_responses[-1])

                # Add tool responses to messages
                messages.extend(tool_responses)

                # Continue conversation with tool results (stream again)
                result = context.lm_studio_client.chat_completion(
                    messages,
                    tools=tools,
                    stream=stream,
                    on_chunk_callback=output_chunk,
                    usage_callback=output_usage,
                )
                # TODO: Remove? Use for a while and check behaviour
                print()  # Newline after streaming

                messages.append(
                    {
                        "role": "assistant",
                        "content": result["content"],
                        **(
                            {"tool_calls": result["tool_calls"]}
                            if result["tool_calls"]
                            else {}
                        ),
                    }
                )

                # Save assistant response to session
                context.session_handler.save_message(messages[-1])

                # Get tool calls from result for next iteration
                tool_calls = result["tool_calls"]

        except KeyboardInterrupt:
            break
        # except Exception as e:
        #     print(f"Cli::shell(): Error: {e}")


if __name__ == "__main__":
    cli()
