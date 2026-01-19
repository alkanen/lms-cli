# lms-cli

A CLI-based code assistant powered by LM Studio. Provides an interactive conversational interface for working with local language models to analyze, search, and modify code in your workspace.

## Features

- **Rich Terminal UI** - Styled panels, syntax highlighting, and live streaming display
- **Token Usage Tracking** - Visual progress bar showing context window usage
- **Tool System** - AI can read/write files, search, and run tests with permission controls
- **Session Persistence** - Save and resume conversations with history compaction
- **Semantic Search** - Find relevant code using embeddings (optional)
- **File References** - Include file contents inline using `@file.py` syntax
- **Workspace Safety** - Enforces boundaries to prevent access outside your project

## Installation

```bash
# Clone the repository
git clone https://github.com/alkanen/lms-cli.git
cd lms-cli

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `click` - CLI framework
- `enquiries` - Interactive prompts (legacy shell)
- `rich` - Terminal UI with panels, syntax highlighting, and progress bars
- `numpy` - Numerical operations for embeddings
- `pyyaml` - Configuration parsing
- `requests` - HTTP client for LM Studio API
- `python-slugify` - String slugification

## Configuration

Copy and edit the configuration file:

```bash
cp config/config.yaml.example config/config.yaml
```

### config/config.yaml

```yaml
lm_studio:
  base_url: "http://localhost:1234/v1"    # LM Studio server URL
  model: "your-model-name"                 # Model to use
  api_key: null                            # Optional API key
  max_tokens: 64000                        # Context window size
  stream: true                             # Enable streaming responses

embeddings:
  enabled: false                           # Enable semantic search
  model: "text-embedding-nomic-embed-text-v1.5"
  dimension: 768
  index_path: ".code_assistant_index"
  include_paths:                           # Paths to index
    - "src"
    - "lib"
  exclude_paths:                           # Paths to skip
    - ".git"
    - "node_modules"

agent:
  system_message: |
    You are a coding assistant...

tools:
  tools_folder: "lms_cli/tools"
  tools_settings:
    - name: "file_search"
      permission_required: true
    - name: "read_file"
      permission_required: true
    - name: "write_file"
      permission_required: true
```

## Usage

### Start Interactive Shell

```bash
python main.py rich --workspace . --config config/config.yaml
```

### Resume Previous Session

```bash
python main.py rich --resume
```

### Send a Single Prompt

```bash
python main.py rich --prompt "Explain what main.py does"
```

### Initialize Embeddings Index (Optional)

```bash
python main.py init --workspace . --excluded ".git" --excluded "node_modules"
```

### Ask Questions Using Semantic Search

```bash
python main.py ask --query "How does authentication work?" --num-files 5
```

## Shell Features

### Rich UI Components

The shell displays information in styled panels:

- **Status Bar** - Shows session ID, model name, and token usage with a progress bar
- **User Messages** - Blue-bordered panels with attachment indicators
- **Assistant Messages** - Green-bordered panels with live streaming animation
- **Tool Requests** - Yellow-bordered panels showing parameters and code previews
- **Tool Results** - Green (success) or red (failure) result panels
- **Errors** - Red panels with suggestions for resolution

### File References

Include file contents in your messages using `@` syntax:

```
> Please review @src/main.py

> Explain lines 10-50 of @lms_cli/core/context.py:10-50
```

### Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands and shortcuts |
| `/compact` | Summarize conversation to reduce token usage |
| `exit` | Exit the shell |

### History Compaction

For long conversations, compact the history to reduce token usage:

```
> /compact Summarize focusing on the authentication changes
```

## Available Tools

The AI assistant can use these tools (with your permission):

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents with optional line ranges |
| `write_file` | Write or append to files within the workspace |
| `file_search` | List files in workspace, optionally filtered by extension |
| `generate_uuid` | Generate deterministic UUIDs from strings |
| `run_test` | Execute pytest tests and return output |

### Permission System

Tools require explicit permission before execution. When prompted, use these shortcuts:

| Key | Action |
|-----|--------|
| `y` | Yes - Allow this specific execution |
| `a` | Always - Always allow this tool (for current session) |
| `n` | No - Deny the request |
| `s` | Suggest - Provide an alternative behavior |
| `f` | Full - Show full content (for file operations) |

Permissions can be granted at file, folder, or workspace level.

## Project Structure

```
lms-cli/
├── main.py                           # Entry point
├── config/
│   └── config.yaml                   # Configuration
├── lms_cli/
│   ├── cli/
│   │   ├── interface.py              # CLI commands (init, ask, shell, rich)
│   │   └── rich_interface.py         # Rich terminal UI implementation
│   ├── core/
│   │   ├── context.py                # Central context manager
│   │   ├── lm_studio_client.py       # LM Studio API client
│   │   ├── tool_registry.py          # Tool loading and execution
│   │   ├── session_handler.py        # Conversation persistence
│   │   ├── workspace.py              # File system operations
│   │   ├── embedding_manager.py      # Semantic search
│   │   └── file_reference_parser.py  # @file syntax parsing
│   └── tools/
│       ├── read_file.py
│       ├── write_file.py
│       ├── file_search.py
│       ├── generate_uuid.py
│       └── run_test.py
└── tests/                            # Unit tests
```

## How It Works

1. **User Input** - You type a message in the shell (displayed in a blue panel)
2. **File References** - `@file.py` references are expanded to include file contents
3. **API Call** - Message sent to LM Studio with available tools
4. **Streaming** - Response streams live with animated indicator in a green panel
5. **Tool Execution** - Tool requests shown in yellow panels with syntax-highlighted previews
6. **Permission** - You approve/deny tool execution using keyboard shortcuts
7. **Tool Results** - Results displayed in green/red panels based on success/failure
8. **Token Tracking** - Status bar updates with token usage after each response
9. **Persistence** - All messages are saved to `~/.lms-cli/sessions/`

## Requirements

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) running with a loaded model
- Local server enabled in LM Studio (default: `http://localhost:1234`)

## Session Storage

Sessions are stored in `~/.lms-cli/sessions/` as JSONL files. Each session contains:

- Complete message history
- Recent working history (for context management)
- Timestamp-based session IDs

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
