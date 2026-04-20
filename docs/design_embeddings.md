# Design: Embedding Index and Semantic Search

## Purpose

The embedding system gives the LLM a semantically-aware window into the
codebase and any additional document corpora the user chooses to index.
Rather than reading files blindly, the LLM can search by meaning and receive
exactly the chunk (function, class, documentation section) that is relevant to
the task at hand.

The system is entirely optional and additive. When `embeddings.enabled` is
`false` or absent, none of the infrastructure is initialised and the
`search_files` tool is not registered.

---

## Goals and Non-Goals

**Goals**

- Semantic search over workspace files and user-specified external directories.
- Multi-granularity index: chunk level (function / paragraph) and document
  level (whole file), both queryable.
- Language-aware chunking: tree-sitter for source code, structure-aware
  chunking for YAML-based config and orchestration files.
- Efficient document-level summaries: extract signatures + docstrings for code;
  structural skeleton for config files — pass only the outline to the LLM
  rather than the full content.
- Incremental re-indexing: only re-process files that have changed since the
  last index pass.
- Clean abstraction boundary so `SQLiteVectorStore` can be replaced by
  ChromaDB, Qdrant, or pgvector without touching any other module.
- Access control: indexing a path constitutes an explicit read grant; tools
  enforce this automatically.

**Non-Goals**

- `~/.ai-cli/` is not a target for indexing (only for config defaults).
- Automatic startup background indexing is not implemented in phase 1 but the
  code is structured to make it trivial to add (see Startup Indexing note).
- Directory-level (above file-level) summary vectors are out of scope.
- PDF or binary file parsing is out of scope (plain text and source code only).

---

## Configuration

```yaml
# ~/.ai-cli/config.yaml  or  <project>/.ai-cli/config.yaml

embeddings:
  enabled: true

  # Embedding model — served at the same base_url as the LLM by default.
  # Any OpenAI-compatible /v1/embeddings endpoint works (Ollama, OpenAI, etc.)
  model: nomic-embed-text       # required when enabled: true

  # Backend override — null means inherit from llm.base_url / llm.api_key_env.
  # Set explicitly only when the embedding model lives on a different server.
  base_url: ~
  api_key_env: ~

  chunking:
    strategy: auto              # "auto" | "fixed" | "semantic"
    # auto: tree-sitter for supported languages, structure-aware for YAML/TOML,
    #       fixed-size fallback for everything else.
    chunk_size: 1200            # characters; used by FixedSizeChunker
    chunk_overlap: 200          # character overlap between adjacent chunks
    max_file_chunks: 300        # skip files that would produce more chunks than this
    min_chunk_chars: 80         # merge tree-sitter nodes smaller than this
    max_chunk_chars: 3000       # split nodes larger than this at blank-line boundaries

  document_embedding:
    enabled: true
    strategy: auto              # "auto" | "average" | "summary"
    # auto selects:
    #   "average"  for code files (tree-sitter or fixed chunks)
    #   "summary"  for prose files (see prose_extensions below)
    prose_extensions:           # file types treated as prose; get LLM-summary doc vectors
      - .md
      - .txt
      - .rst
      - .adoc
    summary_model: ~            # null = inherit from llm.model
    summary_max_tokens: 400     # caps input text chars sent to LLM (~4 chars/token)
    summary_response_tokens: ~  # word-count hint in prompt; null = chunk_size // 4
```

### Config resolution

`embeddings.base_url` and `embeddings.api_key_env` fall back to `llm.base_url`
and `llm.api_key_env` respectively when `null` or absent.
`ConfigManager.get_embedding_config()` performs this merge and is the sole
point where the LLM config is consulted on behalf of the embedding subsystem.

---

## Storage Layout

All index data for a project lives inside the project workspace:

```
<project>/.ai-cli/
└── embeddings/
    └── index.db          # single SQLite database (WAL mode)
```

The database contains three tables:

```sql
-- Each row is either a chunk or a document-level embedding.
CREATE TABLE chunks (
    id           TEXT PRIMARY KEY,    -- xxhash64(file_path || '\x00' || chunk_type || '\x00'
                                      --   || coalesce(start_line, 'doc')) as 16-char hex;
                                      --   delimiter-free, safe on all platforms
    file_path    TEXT NOT NULL,       -- absolute path
    chunk_type   TEXT NOT NULL,       -- "chunk" | "document"
    start_line   INTEGER,             -- NULL for document-level
    end_line     INTEGER,             -- NULL for document-level
    symbol_name  TEXT,                -- e.g. "AuthManager.authenticate" (tree-sitter only)
    symbol_kind  TEXT,                -- "function" | "method" | "class" | "resource" etc.
    char_hash    TEXT NOT NULL,       -- xxhash64 of the entire file content (NOT the chunk text);
                                      --   all chunks for the same file share this hash value
    indexed_at   TEXT NOT NULL,       -- ISO-8601 UTC timestamp
    vector       BLOB NOT NULL        -- float32 little-endian bytes (np.ndarray.tobytes())
);

CREATE INDEX idx_chunks_file ON chunks(file_path);
CREATE INDEX idx_chunks_type ON chunks(chunk_type);

-- Tracks which roots the user has added for indexing.
-- The workspace root is always present; external paths are user-added.
CREATE TABLE index_roots (
    path         TEXT PRIMARY KEY,    -- absolute path
    label        TEXT,                -- optional user-visible name
    added_at     TEXT NOT NULL        -- ISO-8601 UTC timestamp
);

-- Global metadata for this index.
CREATE TABLE meta (
    key          TEXT PRIMARY KEY,
    value        TEXT NOT NULL
);
-- meta rows: "model", "dimension", "schema_version"
```

The `char_hash` column stores the xxhash64 of the **entire file content**
(via the `xxhash` package), not of the individual chunk's text. All chunks
belonging to the same file share the same `char_hash` value. This is what
`VectorStore.all_file_hashes()` returns — one hash per file — and what
`EmbeddingIndex.index()` compares against to skip unchanged files. Using the
full-file hash (rather than mtime) is reliable across filesystem mounts and
version-control checkouts.

**Human-readability**: The `chunks` and `index_roots` tables are fully readable
via the `sqlite3` CLI. Only the `vector` BLOB column is binary. Example:

```
sqlite3 .ai-cli/embeddings/index.db \
  "SELECT file_path, start_line, end_line, symbol_name, symbol_kind
   FROM chunks WHERE chunk_type='chunk' LIMIT 10;"
```

---

## Module Layout

```
ai_cli/
├── core/
│   ├── embedding_provider.py    # EmbeddingProvider ABC + OpenAIEmbeddingProvider
│   ├── vector_store.py          # VectorStore ABC + SQLiteVectorStore
│   ├── chunker.py               # Chunk, ChunkStrategy ABC, all Chunker implementations,
│   │                            #   make_chunker() factory
│   └── embedding_index.py       # IndexRoot, EmbeddingIndex
│                                #   (orchestrates chunking, embedding, storage, access control)
├── tools/
│   └── search_files.py           # search_files tool (LLM-facing)
└── cli/
    └── repl.py                  # /index slash command added here
```

The embedding subsystem is optional. Dependencies are declared under
`[project.optional-dependencies]` in `pyproject.toml`:

```
pip install ai-cli[embeddings]           # numpy, xxhash, tomli (Python <3.11 only)
pip install ai-cli[embeddings,semantic]  # all of the above + tree-sitter grammars
```

- **`numpy>=1.24`** — vectorised dot-product search in `SQLiteVectorStore` and
  document-vector averaging in `EmbeddingIndex`.
- **`xxhash>=3.0`** — fast non-cryptographic hashing for per-file change detection.
- **`tomli>=2.0; python_version < '3.11'`** — TOML chunking on Python < 3.11
  (`tomllib` is stdlib on 3.11+).
- **tree-sitter grammars** (`[semantic]` extra only) — language-aware chunking.

Users who set `embeddings.enabled: false` (the default) incur no additional
install cost.

---

## `EmbeddingProvider` — ABC and Implementation

```python
from __future__ import annotations
from abc import ABC, abstractmethod

class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        on_batch: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        """Async bulk embed — used by EmbeddingIndex.index().
        on_batch(chunks_done, chunks_total) is called after each batch."""

    @abstractmethod
    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Sync single-query embed — used by search() and the summary path.
        Must NOT use asyncio.run() internally; uses the sync openai client."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the vectors returned by embed() / embed_sync()."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Name of the embedding model in use."""

    async def aclose(self) -> None:
        """Close the async HTTP client. Called after each indexing run so the
        next run starts with a fresh client bound to the new event loop."""
```

`OpenAIEmbeddingProvider` uses the `openai` client library (already a
dependency) hitting the `/v1/embeddings` endpoint. It reads `base_url`,
`api_key`, and `model` from the resolved embedding config. API key resolution
is the sole responsibility of `ConfigManager`; the provider never reads
environment variables directly.

Requests are batched according to the configurable `batch_size` parameter
(default 32; local servers such as LM Studio can stall on large batches, so
the default is kept conservative). Larger input lists are split and
concatenated. Callers may increase `batch_size` for cloud APIs that support
larger batches; a warning is logged when the value exceeds 512. A configurable
`request_timeout` (default 120 s) prevents silent hangs on large batches.

---

## `VectorStore` — ABC and SQLite Implementation

### ABC

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class SearchResult:
    id: str
    score: float            # cosine similarity in [-1, 1]
    metadata: dict          # all non-vector columns from the chunks table

class VectorStore(ABC):
    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict],
    ) -> None:
        """Insert or replace rows. ids, vectors, and metadata are parallel lists."""

    @abstractmethod
    def delete_by_file(self, file_path: str) -> None:
        """Remove all chunks whose file_path matches exactly."""

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        k: int = 10,
        chunk_type: str | None = None,   # None = search all types
        path_glob: str | None = None,    # SQLite GLOB pattern against file_path
    ) -> list[SearchResult]:
        """Return the k most similar results by cosine similarity."""

    @abstractmethod
    def all_file_hashes(self) -> dict[str, str]:
        """Return {file_path: char_hash} for every file that has any chunk.
        Used by EmbeddingIndex to detect changed and deleted files efficiently."""

    @abstractmethod
    def clear(self) -> None:
        """Delete all rows from the chunks table."""

    def close(self) -> None:
        """Release resources. No-op on the base class."""
```

### `SQLiteVectorStore`

Stores vectors as `float32` blobs. File paths are stored as POSIX strings
(`path.as_posix()`) for cross-platform consistency. Stored vectors are
L2-normalised row-wise on upsert; cosine similarity then reduces to a dot
product with the L2-normalised query vector. Search loads all matching vectors
into numpy and performs the dot product in one vectorised call — fast enough
for <200 k chunks (sub-100 ms on CPU). No separate norms cache is needed.

**Thread safety**: all DB operations are protected by a `threading.Lock`.
The connection is created with `check_same_thread=False`. This ensures the
store is safe to use across threads, including the background indexing thread
and the main thread (search / tool calls). `OpenAIEmbeddingProvider` also
uses a `threading.Lock` to guard lazy client construction and `_dimension`
writes, since `embed_sync()` can be called concurrently from the main thread
(search) and from `asyncio.to_thread` (summary embedding during indexing).

Writes use WAL mode (`PRAGMA journal_mode=WAL`) for atomic updates during
incremental indexing. All upserts for a single file are wrapped in one
transaction.

**Migration path**: to migrate to a proper vector database, implement
`VectorStore` on the new backend, write a one-time migration script that calls
`old_store.export()` → `new_store.upsert()`, and update the factory function
in `embedding_index.py`. No other code changes are required.

---

## Chunker — Strategy Pattern

### `Chunk` dataclass

```python
@dataclass
class Chunk:
    start_line: int           # 1-based, inclusive
    end_line: int             # 1-based, inclusive
    text: str                 # the chunk's source text
    symbol_name: str | None = None   # fully-qualified name if tree-sitter
    symbol_kind: str | None = None   # "function" | "method" | "class" | ...
```

### `ChunkStrategy` ABC

```python
class ChunkStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str, path: Path) -> list[Chunk]:
        """Split text into chunks. path is used only for language detection."""
```

### `FixedSizeChunker`

Slides a window of `chunk_size` characters with `chunk_overlap` overlap,
aligned to line boundaries. The final window is merged into the previous one
if it is shorter than `min_chunk_chars`.

### `TreeSitterChunker`

Raises `ImportError` at construction time if `tree-sitter` is not installed,
so `make_chunker()` can catch it and fall back gracefully.

Supported languages and their top-level node types:

| Extension(s) | Language | Chunked node types |
|---|---|---|
| `.py` | Python | `function_definition`, `class_definition`, `decorated_definition` |
| `.cpp` `.cc` `.cxx` `.h` `.hpp` | C/C++ | `function_definition`, `class_specifier`, `namespace_definition` |
| `.rs` | Rust | `function_item`, `impl_item`, `struct_item`, `trait_item`, `mod_item` |
| `.lua` | Lua | `function_declaration`, `local_function`, `function_definition` |
| `.go` | Go | `function_declaration`, `method_declaration`, `type_declaration` |
| `.js` | JavaScript | `function_declaration`, `class_declaration`, `method_definition` |
| `.ts` | TypeScript | same + `interface_declaration`, `type_alias_declaration` |
| `.sh` `.bash` | Bash | `function_definition` |

Edge-case handling:

- Nodes shorter than `min_chunk_chars`: collect consecutive short siblings and
  emit them as one chunk labelled with the first node's name.
- Nodes longer than `max_chunk_chars`: split at blank-line boundaries within
  the node body, keeping the node header (signature line) in every sub-chunk.

### Domain Chunkers for Configuration Formats

These use PyYAML's `yaml` module (via the `pyyaml` dependency declared in
`pyproject.toml`) or simple heuristics. No packages beyond PyYAML are required
for YAML-based chunkers. TOML chunking requires `tomllib` (Python 3.11+ stdlib)
or `tomli` on Python < 3.11 (included in the `[embeddings]` optional extra).

#### `MultiDocYamlChunker` — Kubernetes / generic multi-document YAML

Splits on `---` document separators using `yaml.safe_load_all()` from PyYAML. Each
document is one chunk. The `symbol_name` is `{kind}/{metadata.name}` when
those keys are present, falling back to the document index. Suitable for raw
Kubernetes manifests and any other multi-document YAML files.

#### `AnsibleChunker`

Detects Ansible playbooks by the presence of a top-level list where items have
a `hosts` key. Chunks at the **task** level: each item in a play's `tasks`,
`pre_tasks`, or `post_tasks` list is one chunk, labelled with the task's
`name` field. The play itself is also emitted as a document-level chunk.

#### `ComposeChunker`

Detects `docker-compose` files by filename pattern. Each entry under
`services:` is one chunk, labelled with the service name. Handles unquoted
(`web:`), single-quoted (`'web':`), and double-quoted (`"web":`) service keys.
Falls back to a single services-block chunk when no individual service keys can
be matched in the source text (e.g. keys using characters outside `[A-Za-z0-9_.-]`).

#### `TomlChunker`

Splits at top-level table headers (`[section]`) using `tomllib` (Python 3.11+
stdlib) or `tomli` on Python < 3.11 (included in the `[embeddings]` optional
extra). Each section is one chunk.

#### Helm charts

- `Chart.yaml`: single-chunk (whole file is a natural unit).
- `values.yaml`: `FixedSizeChunker` or top-level-key splitting (plain YAML,
  no Go templates).
- `templates/*.yaml`: Go template directives (`{{ }}`) make these invalid YAML.
  Strip template tags with a regex before attempting any parse, or fall back
  directly to `FixedSizeChunker`. Do not emit parse errors to the user.

### `make_chunker()` — Factory

```python
def make_chunker(path: Path, config: dict, text: str | None = None) -> ChunkStrategy:
    strategy = config.get("strategy", "auto")

    if strategy == "fixed":
        return FixedSizeChunker(config)

    # Domain chunkers — checked before tree-sitter (filename/content-based detection)
    if _is_helm_template(path):
        return FixedSizeChunker(config)              # graceful fallback
    if _is_multi_doc_yaml(path):
        return MultiDocYamlChunker(config)
    if _is_ansible_playbook(path):
        return AnsibleChunker(config)
    if _is_compose_file(path):
        return ComposeChunker(config)
    if path.suffix.lower() == ".toml":
        return TomlChunker(config)

    # Tree-sitter for known source languages
    if strategy in ("auto", "semantic"):
        ext = path.suffix.lower()
        if ext in _TREESITTER_LANGUAGES:
            try:
                return TreeSitterChunker(ext, config)
            except ImportError:
                pass   # tree-sitter not installed — fall through

    return FixedSizeChunker(config)
```

---

## Document-Level Embeddings

After all chunks for a file are embedded, the `EmbeddingIndex` creates (or
updates) a single document-level row (`chunk_type = "document"`).

### Strategy selection

```
auto:
    file extension in prose_extensions  →  "summary"  (LLM call)
    otherwise                           →  "average"   (no LLM call)
```

### `average`

The document vector is the element-wise mean of all chunk vectors, then
L2-normalised. No API calls. Suitable for code files where all parts carry
useful information.

### `summary`

For prose files and documentation:

1. **Truncate input**: the file text is truncated to `summary_max_tokens * 4`
   characters (approximately 4 chars per token) before being sent to the LLM.
   This bounds token cost; `summary_max_tokens` defaults to 400 (≈ 1600 chars).
2. **LLM summarisation**: call `llm_client.send()` with a fixed system prompt
   and the truncated text, collecting all `"text"` deltas into a single summary
   string. The user prompt includes a word-count hint derived from
   `summary_response_tokens` (default: `chunk_size // 4`) so the LLM targets a
   length that fits within one embedding chunk.
3. **Embed the summary**: call `EmbeddingProvider.embed_sync([summary])`.

No per-call `max_tokens` cap is set at the API level — the LLM client's
configured `max_response_tokens` still applies. This avoids starving reasoning
models of their token budget; the prompt word-count hint provides soft guidance
instead.

LLM summary calls are made **during `/index`** (dispatched via
`asyncio.to_thread` to avoid blocking the event loop), not at query time.
Failed summary calls (e.g. model unavailable, empty response) fall back to
`average` with a warning logged. If `llm_client` is `None`, a clear warning is
logged and the strategy falls back to `average` immediately. The `llm_client`
is the same instance used for the main REPL — no separate client is
constructed.

Strategy resolution is handled by `_resolve_doc_strategy()`, shared by both
`_compute_doc_vector()` and `_embed_and_upsert_file()`. The latter uses it to
decide whether to dispatch to `asyncio.to_thread` — only the `summary` path
(which makes blocking network calls) gets the thread handoff; the `average`
path (pure numpy) runs inline.

The document vector is computed **before** `delete_by_file()` so that the
previous index data stays intact on failure and the window where the file is
absent from the index is reduced to just the delete + upsert store operations.

**`summary_model`**: if set to a non-null value, a warning is logged and the
main LLM client is used anyway. Per-call model switching is not supported by
the `LLMClient` interface; full `summary_model` support would require
constructing a dedicated client and is deferred.

**`summary_max_tokens`**: controls the input character budget (≈ tokens × 4),
not the response length.

**`summary_response_tokens`**: controls the word-count hint injected into the
summary prompt (default: `chunk_size // 4`). This is a soft hint only — no API
cap is applied. `prose_extensions` entries are normalised to lowercase for
case-insensitive matching; a bare YAML string is treated as a single-item list.
Invalid or null `strategy` values default to `"auto"` with a warning.

> **Note:** The "skeleton extraction" optimisation (passing only signatures +
> docstrings for code files, or structural outlines for config files) described
> in earlier design iterations was not implemented. All file types receive the
> full (truncated) text. Skeleton extraction may be added as an enhancement.

---

## `EmbeddingIndex` — Orchestration Layer

```python
@dataclass
class IndexRoot:
    path: Path
    label: str | None
    added_at: str           # ISO-8601 UTC

class EmbeddingIndex:
    def __init__(
        self,
        db_path: Path,
        provider: EmbeddingProvider,
        store: VectorStore,
        config: dict,
        workspace: Workspace,                  # provides is_ignored() and workspace root path
        llm_client: LLMClient | None = None,   # needed for summary strategy
    ) -> None: ...

    # --- Index lifecycle ---

    async def index(
        self,
        roots: list[Path] | None = None,   # None = all known index_roots
        *,
        incremental: bool = True,           # False = full re-index regardless of hashes
        on_progress: Callable[[int, int, str], None] | None = None,
        cancelled: threading.Event | None = None,
    ) -> IndexStats:
        """
        Walk each root, chunk and embed changed files, delete chunks for
        removed files. Respects the workspace IgnoreFilter for the workspace
        root; no ignore filtering for external roots (the user chose to index them).
        on_progress(files_done, files_total, current_file) is called after each file.
        cancelled is checked between embedding batches for cooperative cancellation.
        Returns a summary: files_indexed, files_skipped, files_deleted, chunks_added.
        """

    async def index_file(self, file_path: Path, *, full: bool = False) -> None:
        """Index a single file directly, bypassing root scanning."""

    async def aclose(self) -> None:
        """Close the provider's async HTTP client after an indexing run."""

    # --- Root management ---

    def add_root(self, path: Path, label: str | None = None) -> None:
        """Register an external directory for indexing. Persists to index_roots table.
        Raises ValueError for the filesystem root or a non-existent path."""

    def update_root_label(self, path: Path, label: str | None) -> None:
        """Update the label of an existing root without changing added_at."""

    def remove_root(self, path: Path) -> None:
        """Unregister a root and delete all its chunks from the index."""

    @property
    def roots(self) -> list[IndexRoot]:
        """All known index roots, loaded from the database."""

    @property
    def db_path(self) -> Path:
        """Path to the SQLite database file."""

    # --- Access control ---

    def is_indexed_path(self, path: Path) -> bool:
        """
        Return True if path is at or below any known index root
        (other than the workspace root — that is handled by Workspace.contains()).
        Uses Path.resolve().relative_to() — not string prefix matching.
        Used by read_file to grant access to external indexed paths.
        """

    # --- Search ---

    def search(
        self,
        query: str,
        k: int = 10,
        level: str = "chunk",       # "chunk" | "document" | "both"
        path_glob: str | None = None,
    ) -> list[SearchResult]:
        """
        Embed the query via embed_sync(), run cosine dot-product search,
        filter by path_glob if given, return ranked results.
        """
```

`search()` is synchronous because tool `execute()` methods are synchronous and
are called from within the REPL's running event loop. Using `asyncio.run()`
inside an already-running loop would raise `RuntimeError`. Instead,
`EmbeddingProvider` exposes two paths:

- **`embed()` (async)** — used by `EmbeddingIndex.index()` for bulk indexing;
  called with `await` from the REPL's async context.
- **`embed_sync()` (sync)** — used by `EmbeddingIndex.search()` for single-query
  embedding at tool call time, and by the `summary` document-embedding path
  (dispatched via `asyncio.to_thread` during indexing); uses the synchronous
  `openai.Client` (not `openai.AsyncClient`) under the hood. Thread-safe via
  the provider's `threading.Lock`.

Both methods share the same batching and retry logic; `embed_sync()` is not a
thin `asyncio.run()` wrapper.

### Change Detection

Before indexing a file, `EmbeddingIndex` computes `xxhash64(file_content)` and
compares it against the stored `char_hash`. If they match, the file is skipped.
This is correct across all filesystems and version-control checkouts regardless
of mtime accuracy.

### Startup Indexing (future hook point)

The `index()` method is an `async` coroutine from the start. Adding startup
background indexing later requires only:

```python
# In REPL.__init__ or startup sequence, when index_on_startup: true:
asyncio.create_task(embedding_index.index())
```

No changes to `EmbeddingIndex` itself.

### IgnoreFilter integration

When walking the workspace root, `EmbeddingIndex` uses `workspace.is_ignored()`
to skip files and prune directories, identical to how `find_files` works.
External roots are walked without ignore filtering — the user explicitly chose
to index them.

---

## Access Control: Indexed Roots as Read Grants

Running `/index /path/to/corpus` is an explicit user action that grants the
system read access to that path. The set of index roots is therefore the access
control list for paths outside the workspace.

Both `read_file` and `find_files` check this before acting on a path:

```python
def _is_allowed_path(self, path: Path) -> bool:
    if self._workspace.contains(path):
        return True
    ei = self._workspace.embedding_index
    if ei is not None and ei.is_indexed_path(path):
        return True
    return False
```

The two tools handle a `False` return differently:

- `read_file` calls `workspace.embedding_index.is_indexed_path(path)` before
  potentially entering the normal permission-prompt flow (depending on config).
- `find_files` is permission-free: it validates any optional `path` parameter
  and only allows values that are within the workspace or an indexed external
  root; out-of-scope paths are rejected/denied rather than prompting.

If `embedding_index` is `None` (embeddings disabled or not yet initialised),
the fallback is unchanged: `read_file` may prompt, `find_files` rejects.

This means:
- No symlinks, no special tools, no separate permission config.
- Removing a root via `/index remove /path` revokes read access immediately.
- The SQLite `index_roots` table is the authoritative access-control record
  for external paths. It is loaded at `EmbeddingIndex` construction time and
  persists across process restarts.

---

## `search_files` Tool

```
Tool name:   search_files
Permission:  not required (read-only, searches an already-indexed corpus)

Arguments:
  query       string   required   Natural language or code snippet to search for
  k           int      optional   Number of results to return (default 5, max 20)
  level       string   optional   "chunk" (default) | "document" | "both"
  path_glob   string   optional   Restrict results to files matching this pattern
                                  e.g. "src/**/*.py" or "/home/user/docs/**/*.md"

Returns (canonical success wrapper):
  data:
    results: [
      {
        file:        string     absolute path
        start_line:  int|null   null for document-level results
        end_line:    int|null
        symbol_name: string|null
        symbol_kind: string|null
        score:       float      cosine similarity in [-1.0, 1.0] (higher = more similar)
        snippet:     string     live content of [start_line, end_line] at query time
      }
    ]
    query_time_ms: int
```

The `snippet` field is read live from the file at query time (not stored in the
index), so it always reflects the current content even if the file has changed
since the last `/index` run. The display layer flags stale results (hash
mismatch) with a visual indicator.

`DISABLED_BY_DEFAULT = True` — only available when explicitly enabled via
`tool_manager` or session/persistent enable, same as other non-essential tools.

---

## `/index` Slash Command

```
/index [path] [--label <name>] [--file <path>] [--full] [--remove]
```

| Form | Effect |
|---|---|
| `/index` | Incremental index of all known roots |
| `/index src` | Incremental index of `<workspace>/src` (relative paths resolve against workspace root) |
| `/index /abs/path/to/docs` | Add external root + index it; persists across sessions |
| `/index /abs/path --label corpus` | As above, with a human-readable label |
| `/index --label new-name /abs/path` | Update the label of an existing root |
| `/index --file src/main.py` | Index a single file directly (relative to workspace root) |
| `/index --full` | Force full re-index (ignore hash cache) of all roots |
| `/index --remove ./src` | Remove a root and delete its chunks |

Relative paths in all forms are resolved against the workspace root, not the
process working directory. Tab completion is provided for paths and flags.

The command runs `await embedding_index.index(...)` and reports `IndexStats`
to the display (files indexed / skipped / deleted, time taken).

External roots are persisted in the `index_roots` table and loaded at startup,
so they survive across sessions without any extra configuration.

---

## Integration with `Workspace`

`Workspace` gains one optional attribute:

```python
class Workspace:
    embedding_index: EmbeddingIndex | None  # None when embeddings disabled
```

Set by the startup sequence after `EmbeddingIndex` is constructed:

```python
# In __main__.py startup:
if config.get("embeddings", {}).get("enabled"):
    workspace.embedding_index = build_embedding_index(workspace, config)
```

Tools receive `Workspace` at construction time and access
`workspace.embedding_index` when needed. No tool imports `EmbeddingIndex`
directly.

---

## Dependency Flow (additions only)

```
embedding_index → embedding_provider
embedding_index → vector_store
embedding_index → chunker
embedding_index → workspace          (for is_ignored(), workspace root path)
embedding_index → llm_client         (optional; only for summary strategy)
search_files     → embedding_index    (via workspace.embedding_index)
read_file       → embedding_index    (via workspace.embedding_index; access control)
find_files      → embedding_index    (via workspace.embedding_index; access control)
repl            → embedding_index    (via workspace.embedding_index; /index command)
```

No cycles are introduced. `embedding_index` does not depend on any tool or on
the REPL.

---

## Testing Strategy

### `test_embedding_provider.py`

- Mock the `openai` client; verify `embed()` batches correctly according to
  `batch_size` and concatenates results in order.
- Verify `dimension` is read from the first API response and cached.

### `test_vector_store.py`

- `SQLiteVectorStore`: round-trip upsert → search → delete_by_file.
- Verify cosine similarity ordering with known vectors.
- Verify `all_file_hashes()` returns correct mapping.
- Verify WAL mode is enabled after construction.
- Verify that after `delete_by_file`, no rows with that path remain.

### `test_chunker.py`

- `FixedSizeChunker`: verify chunk boundaries, overlap, min-merge, final-merge.
- `TreeSitterChunker` (skipped if tree-sitter not installed via
  `pytest.importorskip`): parse a small synthetic Python file; verify chunk
  names and line ranges.
- `MultiDocYamlChunker`: two-document YAML → two chunks with correct
  `symbol_name`.
- `AnsibleChunker`: synthetic playbook → task-level chunks with names.
- `make_chunker()`: verify correct strategy is selected for each extension;
  verify fallback when tree-sitter missing.

### `test_embedding_index.py`

- `add_root` / `remove_root`: verify database persistence.
- `is_indexed_path`: workspace root always False (handled by Workspace);
  path under added external root → True; unrelated path → False.
- `index()` incremental: file unchanged (same hash) → not re-embedded;
  file changed → re-embedded; file deleted → chunks removed.
- `index()` full (`incremental=False`): re-embeds all files regardless of hash.
- `search()`: mock provider returns deterministic vectors; verify ranking order.
- Startup indexing hook: verify `index()` is `async` and can be called via
  `asyncio.create_task()` without modification.

### `test_search_files.py`

- Tool schema matches canonical format.
- `execute()` with mock `EmbeddingIndex`: correct args forwarded, results
  returned in canonical wrapper.
- `path_glob` filter applied before returning results.
- Stale file (hash mismatch between index and current content) flagged in
  snippet.

### `test_access_control.py` (additions to existing tool tests)

- `read_file` with path under external indexed root → allowed.
- `read_file` with path outside all roots and `embedding_index is None` →
  existing permission prompt flow unchanged.
- `read_file` with `embedding_index` set but path not under any root →
  permission prompt flow.

---

## Implementation Status

All planned embedding components are implemented and tested (1037 tests, all passing).

| Step | Module | Status |
|---|---|---|
| 1 | `chunker.py` | ✅ |
| 2 | `vector_store.py` | ✅ |
| 3 | `embedding_provider.py` | ✅ |
| 4 | `embedding_index.py` + `Workspace.embedding_index` | ✅ |
| 5 | `read_file` access control for external paths | ✅ |
| 5 | `find_files` access control for `path` parameter | 🔲 planned |
| 6 | `search_files.py` tool | ✅ |
| 7 | `/index` slash command in `repl.py` + tab completion | ✅ |
| 8 | `ConfigManager.get_embedding_config()` | ✅ |
| + | `summary` document-embedding strategy | ✅ |
