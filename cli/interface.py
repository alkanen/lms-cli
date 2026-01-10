import click

import json

from core.embedding_manager import EmbeddingManager
from core.file_reference_parser import FileReferenceParser
from core.lm_studio_client import LMStudioClient
from core.tool_registry import ToolRegistry
from core.workspace import Workspace


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
    chunks = []
    for chunk in lm_client.chat_completion(messages, stream=True):
        click.echo(chunk, nl=False)
        chunks.append(chunk)

    if chunks:
        messages.append(
            {
                "role": "assistant",
                "content": "".join(chunks),
            }
        )


@cli.command()
@click.option(
    "--config", help="Optional configuration file", default="config/config.yaml"
)
def shell(config):
    """Interactive shell mode"""
    workspace = "."
    lm_client = LMStudioClient(config_path=config)
    tool_registry = ToolRegistry(config_path=config, workspace=workspace)
    tool_registry.load_from_config()

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

            # Stream the response while collecting full message
            chunks = []
            tool_chunks = []

            # First stream chunks to display them immediately
            for partial in lm_client.chat_completion(
                messages, tools=tools, stream=True
            ):
                if "chunk" in partial:
                    chunk = partial["chunk"]
                    click.echo(chunk, nl=False)
                    chunks.append(chunk)

                if "tool_chunks" in partial:
                    tool_chunks.append(partial["tool_chunks"])

            tool_calls = ToolRegistry.process_tool_chunks(tool_chunks)

            # Add to messages with all fields (including tool_calls if present)
            messages.append(
                {
                    "role": "assistant",
                    "content": "".join(chunks),
                    **(
                        {
                            "tool_calls": tool_calls
                        }
                        if tool_calls else {}
                    ),
                }
            )

            # Check for tool calls
            if tool_calls:
                tool_calls = lm_client.parse_tool_calls(
                    {"choices": [{"message": {"tool_calls": tool_calls}}]}
                )

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
                chunks = []
                tool_chunks = []

                for partial in lm_client.chat_completion(
                    messages, tools=tools, stream=True
                ):
                    if "chunk" in partial:
                        chunk = partial["chunk"]
                        click.echo(chunk, nl=False)
                        chunks.append(chunk)

                    if "tool_chunks" in partial:
                        tool_chunks.append(partial["tool_chunks"])

                tool_calls = ToolRegistry.process_tool_chunks(tool_chunks)

                messages.append(
                    {
                        "role": "assistant",
                        "content": "".join(chunks),
                        **(
                            {
                                "tool_calls": tool_calls
                            }
                            if tool_calls else {}
                        ),
                    }
                )
                if tool_calls:
                    # Not sure if anything should be done with this...
                    tool_calls = lm_client.parse_tool_calls(
                        {"choices": [{"message": {"tool_calls": tool_calls}}]}
                    )

        except KeyboardInterrupt:
            break
        # except Exception as e:
        #     print(f"Cli::shell(): Error: {e}")


if __name__ == "__main__":
    cli()
