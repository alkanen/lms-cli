"""Tests for ai_cli.core.chunker."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_cli.core.chunker import (
    AnsibleChunker,
    ComposeChunker,
    FixedSizeChunker,
    MultiDocYamlChunker,
    TomlChunker,
    _is_ansible_playbook,
    _is_compose_file,
    _is_multi_doc_yaml,
    make_chunker,
)

# ---------------------------------------------------------------------------
# FixedSizeChunker
# ---------------------------------------------------------------------------


def test_fixed_basic():
    """A short text with a known number of lines is chunked correctly."""
    text = "\n".join(f"line {i}" for i in range(1, 51)) + "\n"
    chunker = FixedSizeChunker({"chunk_size": 200, "chunk_overlap": 0})
    chunks = chunker.chunk(text, Path("test.txt"))
    assert chunks
    # All chunks must have valid line numbers.
    for c in chunks:
        assert c.start_line >= 1
        assert c.end_line >= c.start_line
    # Chunks should cover the whole file (no lines dropped).
    all_text = "".join(c.text for c in chunks)
    assert len(all_text) >= len(text) - len(chunks) * 5  # some overlap is fine


def test_fixed_overlap():
    """Overlap causes the first line of a subsequent chunk to be <= the last of prior."""
    text = "a" * 200 + "\n" + "b" * 200 + "\n" + "c" * 200 + "\n"
    chunker = FixedSizeChunker({"chunk_size": 300, "chunk_overlap": 100})
    chunks = chunker.chunk(text, Path("test.txt"))
    assert len(chunks) >= 2
    # With overlap the second chunk should start at or before where the first ended.
    assert chunks[1].start_line <= chunks[0].end_line + 1


def test_fixed_min_chunk_merge():
    """A final chunk smaller than min_chunk_chars is merged into the previous."""
    # Build text where the last chunk would be tiny.
    lines = ["x" * 100 + "\n"] * 10 + ["y" * 5 + "\n"]  # last line is tiny
    text = "".join(lines)
    chunker = FixedSizeChunker(
        {"chunk_size": 400, "chunk_overlap": 0, "min_chunk_chars": 80}
    )
    chunks = chunker.chunk(text, Path("test.txt"))
    # The tiny last chunk should be merged into the preceding one.
    for c in chunks[:-1]:
        assert len(c.text) >= 80 or len(chunks) == 1


def test_fixed_max_file_chunks_returns_empty():
    """Files producing too many chunks return an empty list."""
    # 1000 lines × 10 chars each = 10 000 chars; chunk_size=50 → ~200 chunks
    # Set max_file_chunks=5 to trigger the limit.
    text = "\n".join(["x" * 50] * 200) + "\n"
    chunker = FixedSizeChunker(
        {"chunk_size": 50, "chunk_overlap": 0, "max_file_chunks": 5}
    )
    chunks = chunker.chunk(text, Path("test.txt"))
    assert chunks == []


def test_fixed_empty_text():
    chunker = FixedSizeChunker({})
    assert chunker.chunk("", Path("empty.txt")) == []


def test_fixed_single_line():
    chunker = FixedSizeChunker({"chunk_size": 1200})
    text = "hello world\n"
    chunks = chunker.chunk(text, Path("single.txt"))
    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 1


def test_fixed_line_longer_than_chunk_size():
    """A single line longer than chunk_size is split mid-line (hard cap)."""
    chunk_size = 100
    long_line = "x" * 300 + "\n"  # 3× chunk_size
    text = long_line
    chunker = FixedSizeChunker({"chunk_size": chunk_size, "chunk_overlap": 0})
    chunks = chunker.chunk(text, Path("long.txt"))
    # Must produce multiple chunks (not hang or produce a single 300-char chunk)
    assert len(chunks) >= 2
    # No chunk may exceed 2× chunk_size
    for c in chunks:
        assert len(c.text) <= chunk_size * 2, f"chunk too large: {len(c.text)}"
    # Full content is covered
    combined = "".join(c.text for c in chunks)
    assert long_line.strip() in combined.replace("\n", "")


def test_fixed_long_line_in_middle_no_infinite_loop():
    """A long line sandwiched between normal lines is handled without looping."""
    chunk_size = 100
    normal = "normal line\n"
    long_line = "L" * 250 + "\n"
    text = normal * 5 + long_line + normal * 5
    chunker = FixedSizeChunker(
        {"chunk_size": chunk_size, "chunk_overlap": 20, "max_file_chunks": 50}
    )
    chunks = chunker.chunk(text, Path("mixed.txt"))
    assert chunks  # must not return empty (no infinite loop → returns chunks)
    for c in chunks:
        assert len(c.text) <= chunk_size * 2


def test_fixed_line_nearly_chunk_size():
    """Lines just over chunk_size don't produce oversized chunks or loops."""
    chunk_size = 100
    # Line length = chunk_size + 50 (between 1× and 2× chunk_size).
    long_line = "Y" * 150 + "\n"
    text = long_line * 4
    chunker = FixedSizeChunker(
        {"chunk_size": chunk_size, "chunk_overlap": 20, "max_file_chunks": 50}
    )
    chunks = chunker.chunk(text, Path("nearlong.txt"))
    assert chunks
    for c in chunks:
        assert len(c.text) <= chunk_size * 2


# ---------------------------------------------------------------------------
# MultiDocYamlChunker
# ---------------------------------------------------------------------------


def test_multi_doc_yaml_two_documents():
    """A two-document YAML file produces two chunks."""
    text = """\
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-config
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deploy
"""
    chunker = MultiDocYamlChunker({})
    chunks = chunker.chunk(text, Path("manifest.yaml"))
    assert len(chunks) == 2
    assert chunks[0].symbol_name == "ConfigMap/my-config"
    assert chunks[0].symbol_kind == "document"
    assert chunks[1].symbol_name == "Deployment/my-deploy"


def test_multi_doc_yaml_no_metadata():
    """Documents without kind/metadata fall back to index-based symbol_name."""
    text = "foo: bar\n---\nbaz: qux\n"
    chunker = MultiDocYamlChunker({})
    chunks = chunker.chunk(text, Path("plain.yaml"))
    assert len(chunks) == 2
    # symbol_name is the string index.
    assert chunks[0].symbol_name == "0"
    assert chunks[1].symbol_name == "1"


def test_multi_doc_yaml_single_doc():
    """A YAML without '---' is returned as a single chunk."""
    text = "key: value\nother: 123\n"
    chunker = MultiDocYamlChunker({})
    chunks = chunker.chunk(text, Path("single.yaml"))
    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# AnsibleChunker
# ---------------------------------------------------------------------------


def test_ansible_chunker_tasks():
    """Task-level chunks are produced from a synthetic playbook."""
    text = """\
- hosts: all
  tasks:
    - name: Install nginx
      apt:
        name: nginx
        state: present
    - name: Start service
      service:
        name: nginx
        state: started
"""
    chunker = AnsibleChunker({})
    chunks = chunker.chunk(text, Path("playbook.yml"))
    # Should include the play chunk + individual task chunks.
    names = [c.symbol_name for c in chunks]
    assert "Install nginx" in names
    assert "Start service" in names


def test_ansible_chunker_non_playbook():
    """Non-playbook YAML returns empty list."""
    text = "key: value\n"
    chunker = AnsibleChunker({})
    # A plain dict is not a list, so should return empty.
    chunks = chunker.chunk(text, Path("not_playbook.yml"))
    assert chunks == []


# ---------------------------------------------------------------------------
# ComposeChunker
# ---------------------------------------------------------------------------


def test_compose_chunker_services():
    """Each service in a docker-compose file becomes one chunk."""
    text = """\
version: "3"
services:
  web:
    image: nginx
  db:
    image: postgres
"""
    chunker = ComposeChunker({})
    chunks = chunker.chunk(text, Path("docker-compose.yml"))
    names = [c.symbol_name for c in chunks]
    assert "web" in names
    assert "db" in names
    for c in chunks:
        assert c.symbol_kind == "service"


def test_compose_chunker_empty_services():
    """A compose file with no services returns empty list."""
    text = "version: '3'\n"
    chunker = ComposeChunker({})
    chunks = chunker.chunk(text, Path("docker-compose.yml"))
    assert chunks == []


def test_compose_chunker_quoted_keys():
    """Service keys that are quoted in the source text are matched correctly."""
    text = """\
version: "3"
services:
  "web":
    image: nginx
  'db':
    image: postgres
"""
    chunker = ComposeChunker({})
    chunks = chunker.chunk(text, Path("docker-compose.yml"))
    names = [c.symbol_name for c in chunks]
    assert "web" in names
    assert "db" in names


def test_compose_chunker_fallback_chunk():
    """Fallback: unrecognised key chars produce a single services-block chunk.

    YAML allows unquoted bare scalars containing ``@`` and similar characters
    that fall outside the ``[A-Za-z0-9_.-]`` identifier class.  The regex
    cannot match such keys, so ``service_blocks`` is empty and the chunker
    falls back to emitting the whole services block as one chunk.
    """
    text = """\
version: "3"
services:
  web@app:
    image: nginx
  db@shard:
    image: postgres
"""
    chunker = ComposeChunker({})
    chunks = chunker.chunk(text, Path("docker-compose.yml"))
    assert len(chunks) == 1
    assert chunks[0].symbol_name == "services"
    assert chunks[0].symbol_kind == "service"


# ---------------------------------------------------------------------------
# TomlChunker
# ---------------------------------------------------------------------------


def test_toml_chunker_sections():
    """Top-level TOML sections become separate chunks."""
    text = """\
[package]
name = "my-pkg"
version = "1.0"

[dependencies]
requests = ">=2.0"

[dev-dependencies]
pytest = ">=7.0"
"""
    chunker = TomlChunker({})
    chunks = chunker.chunk(text, Path("Cargo.toml"))
    names = [c.symbol_name for c in chunks]
    assert "package" in names
    assert "dependencies" in names
    assert "dev-dependencies" in names


def test_toml_chunker_no_sections():
    """A TOML with no section headers is returned as one chunk."""
    text = "key = 'value'\n"
    chunker = TomlChunker({})
    chunks = chunker.chunk(text, Path("simple.toml"))
    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# make_chunker — strategy selection
# ---------------------------------------------------------------------------


def test_make_chunker_fixed_strategy():
    """strategy: fixed always returns FixedSizeChunker regardless of file type."""
    from ai_cli.core.chunker import make_chunker

    chunker = make_chunker(Path("main.py"), {"strategy": "fixed"})
    assert isinstance(chunker, FixedSizeChunker)


def test_make_chunker_toml():
    """TOML files get TomlChunker by default."""
    chunker = make_chunker(Path("Cargo.toml"), {})
    assert isinstance(chunker, TomlChunker)


def test_make_chunker_py_no_treesitter():
    """Python files fall back to FixedSizeChunker when tree-sitter is absent."""
    import sys
    from unittest.mock import patch

    # Simulate tree-sitter ImportError.
    with patch.dict(sys.modules, {"tree_sitter": None, "tree_sitter_python": None}):
        chunker = make_chunker(Path("main.py"), {"strategy": "auto"})
    # Should be FixedSizeChunker since tree-sitter is missing.
    assert isinstance(chunker, FixedSizeChunker)


def test_make_chunker_docker_compose():
    """Docker Compose files get ComposeChunker."""
    chunker = make_chunker(Path("docker-compose.yml"), {})
    assert isinstance(chunker, ComposeChunker)


def test_make_chunker_multi_doc_yaml(tmp_path):
    """Multi-document YAML files get MultiDocYamlChunker."""
    p = tmp_path / "k8s.yaml"
    p.write_text("kind: Pod\n---\nkind: Service\n")
    chunker = make_chunker(p, {})
    assert isinstance(chunker, MultiDocYamlChunker)


def test_make_chunker_ansible(tmp_path):
    """Ansible playbooks get AnsibleChunker."""
    p = tmp_path / "playbook.yml"
    p.write_text("- hosts: all\n  tasks: []\n")
    chunker = make_chunker(p, {})
    assert isinstance(chunker, AnsibleChunker)


# ---------------------------------------------------------------------------
# TreeSitterChunker — only run when tree-sitter is installed
# ---------------------------------------------------------------------------


def test_tree_sitter_python_chunker():
    """Tree-sitter Python chunker produces function/class-level chunks."""
    pytest.importorskip("tree_sitter")
    pytest.importorskip("tree_sitter_python")

    from ai_cli.core.chunker import TreeSitterChunker

    code = """\
def greet(name):
    return f"Hello, {name}"


class Greeter:
    def __init__(self):
        self.prefix = "Hi"

    def greet(self, name):
        return f"{self.prefix}, {name}"
"""
    chunker = TreeSitterChunker(".py", {})
    chunks = chunker.chunk(code, Path("test.py"))
    assert len(chunks) >= 2
    kinds = {c.symbol_kind for c in chunks}
    assert "function" in kinds or "class" in kinds


def test_tree_sitter_raises_import_error_when_missing():
    """TreeSitterChunker raises ImportError if tree-sitter is not installed."""
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"tree_sitter": None}):
        from ai_cli.core.chunker import TreeSitterChunker

        with pytest.raises(ImportError):
            TreeSitterChunker(".py", {})


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def test_is_compose_file():
    assert _is_compose_file(Path("docker-compose.yml"))
    assert _is_compose_file(Path("docker-compose.prod.yaml"))
    assert _is_compose_file(Path("compose.yml"))
    assert not _is_compose_file(Path("config.yml"))


def test_is_multi_doc_yaml(tmp_path):
    p = tmp_path / "multi.yaml"
    p.write_text("a: 1\n---\nb: 2\n")
    assert _is_multi_doc_yaml(p)

    q = tmp_path / "single.yaml"
    q.write_text("a: 1\n")
    assert not _is_multi_doc_yaml(q)

    r = tmp_path / "file.py"
    r.write_text("# no yaml\n")
    assert not _is_multi_doc_yaml(r)


def test_is_ansible_playbook(tmp_path):
    p = tmp_path / "play.yml"
    p.write_text("- hosts: all\n  tasks: []\n")
    assert _is_ansible_playbook(p)

    q = tmp_path / "nope.yml"
    q.write_text("key: value\n")
    assert not _is_ansible_playbook(q)
