from typing import List, Tuple

import click
import enquiries
import json

from lms_cli.core.embedding_manager import EmbeddingManager
from lms_cli.core.file_reference_parser import FileReferenceParser
from lms_cli.core.lm_studio_client import LMStudioClient
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
    pass


@cli.command()
@click.option("--workspace", default=".", help="Workspace directory")
@click.option("--excluded", multiple=True, help="Folders to exclude")
@click.option(
    "--config", help="Optional configuration file", default="config/config.yaml"
)
def init(workspace, excluded, config):
    """Initialize the workspace and create embeddings index"""
    workspace = Workspace(workspace)
    embedding_manager = EmbeddingManager(config_path=config)

    if not click.confirm("This will create an embedding index of your code. Continue?"):
        return

    included_set = embedding_manager.inclusion_paths
    excluded_set = set(excluded)
    excluded_set.update(embedding_manager.exclusion_paths)

    # Get all files in workspace
    files = workspace.list_files(
        included_folders=included_set, excluded_folders=excluded_set
    )
    print(f"Found {len(files)} files to process")

    # Process files and get embeddings
    lm_client = LMStudioClient(config_path=config)
    embeddings = []
    metadata = []

    embedding_manager.initialize_index()

    for file_path in files:
        content = workspace.read_file(file_path)
        embedding = lm_client.get_embedding(content)

        embeddings.append(embedding)
        filename = str(file_path.relative_to(workspace.root_path))
        metadata.append(
            {
                "file": filename,
                "content": content[:500] + "...",  # Store first 500 chars as preview
            }
        )

    # Add to index
    embedding_manager.add_embeddings(embeddings, metadata)
    print("Embedding index created successfully")


@cli.command()
@click.option("--query", prompt="Enter your request")
@click.option(
    "--num-files", "-n", default=3, help="Maximum number of files to embed in request"
)
@click.option(
    "--config", help="Optional configuration file", default="config/config.yaml"
)
def ask(query, num_files, config):
    """Ask the AI about your code"""
    lm_client = LMStudioClient(config_path=config)
    embedding_manager = EmbeddingManager(config_path=config)

    # First check if we have an index
    try:
        embedding_manager.initialize_index()

        # Get query embedding
        query_embedding = lm_client.get_embedding(query)

        # Search for relevant files
        results = embedding_manager.search(query_embedding, k=num_files)
        print(f"\nFound {len(results)} potentially relevant files:")

        for i, result in enumerate(results, 1):
            print(
                f"{i}. {result['metadata']['file']} "
                f"(similarity: {result['similarity']:.2f})"
            )

        # Prepare context
        context = "\n\n".join(
            f"File: {result['metadata']['file']}\n"
            f"Content:\n{result['metadata']['content']}"
            for result in results
        )

    except Exception as e:
        print(f"Could not use embeddings index: {e}")
        context = "No relevant files found in the embedding index."

    # Prepare messages for chat completion
    messages = [
        {"role": "system", "content": lm_client.system_message},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    # Get response with streaming
    print("\nAssistant:")

    def output_chunk(chunk: str):
        click.echo(chunk, nl=False)

    result = lm_client.chat_completion(
        messages, stream=True, on_chunk_callback=output_chunk
    )
    print()  # Newline after streaming

    if result["content"]:
        messages.append({"role": "assistant", "content": result["content"]})


@cli.command()
@click.option(
    "--config", help="Optional configuration file", default="config/config.yaml")
@click.option(
    "--workspace", help="Path to workspace folder", default="."
)
def shell(config, workspace):
    """Interactive shell mode"""
    def permission_requests(question: str, options: List[str]) -> Tuple[int, str]:
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

    lm_client = LMStudioClient(config_path=config)
    tool_registry = ToolRegistry(
        config_path=config,
        workspace=workspace,
        permission_request_cb=permission_requests,
    )
    tool_registry.load_tools()

    # Prepare initial messages
    messages = [
        {"role": "system", "content": lm_client.system_message},
    ]
    print("Starting interactive shell. Type 'exit' to quit.")

    while True:
        try:
            user_input = input("\n> ").strip()
            if not user_input or user_input.lower() == "exit":
                break

            user_input = FileReferenceParser.parse_message(user_input)

            # Add to messages
            messages.append({"role": "user", "content": user_input})

            # Get available tool definitions
            tools = [
                tool_registry.get_tool_definition(name) for name in tool_registry.tools
            ]

            # Define callback to output chunks as they arrive
            def output_chunk(chunk: str):
                click.echo(chunk, nl=False)

            # Get response with streaming and tool support
            result = lm_client.chat_completion(
                messages, tools=tools, stream=True, on_chunk_callback=output_chunk
            )
            print()  # Newline after streaming

            # Add to messages with all fields (including tool_calls if present)
            messages.append(
                {
                    "role": "assistant",
                    "content": result["content"],
                    **({"tool_calls": result["tool_calls"]} if result["tool_calls"] else {}),
                }
            )

            # Get tool calls from result
            tool_calls = result["tool_calls"]

            while tool_calls:
                # print("\nTool calls detected:")
                # for tc in tool_calls:
                #     print(f"- {tc['function']['name']}({tc['function']['arguments']})")

                # Execute tools and collect responses
                tool_responses = []
                for tc in tool_calls:
                    try:
                        result = tool_registry.execute_tool(
                            tc["function"]["name"], tc["function"]["arguments"]
                        )
                        # print(f"\nTool {tc['function']['name']} returned: {result}")

                        # Add tool response to messages
                        tool_responses.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": json.dumps(result)
                            }
                        )
                    except Exception as e:
                        print(f"Tool call `{tc['function']['name']}({tc['function']['arguments']}) failed: {str(e)}")
                        tool_responses.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": f"Error {str(e)}"
                            }
                        )

                # Add tool responses to messages
                messages.extend(tool_responses)

                # Continue conversation with tool results (stream again)
                print("\nAssistant:")

                result = lm_client.chat_completion(
                    messages, tools=tools, stream=True, on_chunk_callback=output_chunk
                )
                print()  # Newline after streaming

                messages.append(
                    {
                        "role": "assistant",
                        "content": result["content"],
                        **({"tool_calls": result["tool_calls"]} if result["tool_calls"] else {}),
                    }
                )

                # Get tool calls from result for next iteration
                tool_calls = result["tool_calls"]

        except KeyboardInterrupt:
            break
        # except Exception as e:
        #     print(f"Cli::shell(): Error: {e}")


if __name__ == "__main__":
    cli()
