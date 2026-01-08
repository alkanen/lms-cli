import click
from core.lm_studio_client import LMStudioClient
from core.workspace import Workspace
from core.tool_registry import ToolRegistry
from core.embedding_manager import EmbeddingManager


@click.group()
def cli():
    """Interactive CLI code assistant"""
    pass


@cli.command()
@click.option("--workspace", default=".", help="Workspace directory")
@click.option("--excluded", multiple=True, help="Folders to exclude")
def init(workspace, excluded):
    """Initialize the workspace and create embeddings index"""
    workspace = Workspace(workspace)
    embedding_manager = EmbeddingManager()

    if not click.confirm("This will create an embedding index of your code. Continue?"):
        return

    included_set = embedding_manager.inclusion_paths
    excluded_set = set(excluded)
    excluded_set.update(embedding_manager.exclusion_paths)

    # Get all files in workspace
    files = workspace.list_files(included_folders=included_set, excluded_folders=excluded_set)
    print(f"Found {len(files)} files to process")

    # Process files and get embeddings
    lm_client = LMStudioClient()
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
                "content": content[:500] + "...",  # Store first 500 chars aas preview
            }
        )

    # Add to index
    embedding_manager.add_embeddings(embeddings, metadata)
    print("Embedding index created successfully")


@cli.command()
@click.option("--query", prompt="Enter your request")
@click.option("--num-files", "-n", default=3, prompt="Maximum number of files to embed in request")
def ask(query, num_files):
    """Ask the AI about your code"""
    workspace = Workspace(".")
    lm_client = LMStudioClient()
    embedding_manager = EmbeddingManager()

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
                f"{i}. {result['metadata']['file']} (similarity: {result['similarity']:.2f})"
            )

        # Prepare context
        context = "\n\n".join(
            f"File: {result['metadata']['file']}\nContent:\n{result['metadata']['content']}"
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

    # Get response
    response = lm_client.chat_completion(messages)
    print("\nAssistant:", response["choices"][0]["message"]["content"])


@cli.command()
def shell():
    """Interactive shell mode"""
    workspace = Workspace(".")
    lm_client = LMStudioClient()
    tool_registry = ToolRegistry()

    # Prepare messages
    messages = [
        {"role": "system", "content": lm_client.system_message},
    ]
    print("Starting interactive shell. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\n> ").strip()
            if not user_input or user_input.lower() == "exit":
                break

            # Add to messages
            messages.append({"role": "user", "content": user_input})

            # Get available tool definitions
            tools = [
                tool_registry.get_tool_definition(name) for name in tool_registry.tools
            ]

            # First call without tools to see if we need any
            response = lm_client.chat_completion(messages, tools=tools)

            # Check for tool calls
            tool_calls = lm_client.parse_tool_calls(response)

            while tool_calls:
                print("\nTool calls detected:")
                for tc in tool_Calls:
                    print(f"- {tc['function']['name']}({tc['function']['arguments']})")

                # Execute tools
                new_messages = []
                for tc in tool_calls:
                    try:
                        result = tool_registry.execute_tool(
                            tc["function"]["name"], eval(tc["function"]["arguments"])
                        )
                        print(f"\nTool {tc['function']['name']} returned: {result}")

                        # Add tool response to messages
                        new_messages.append(
                            {
                                "role": "tool",
                                "content": str(result),
                                "tool_call_id": tc["id"],
                            }
                        )
                    except Exception as e:
                        new_messages.append(
                            {
                                "role": "tool",
                                "content": f"Error {str(e)}",
                                "tool_call_id": tc["id"],
                            }
                        )

                # Add tool responses to messages
                messages.extend(new_messages)

                # Continue conversation with tool results
                response = lm_client.chat_completion(messages, tools=tools)
                tool_calls = lm_client.parse_tool_calls(response)

            # Print final response
            print("\nAssistant:", response["choices"][0]["message"]["content"])

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    cli()
