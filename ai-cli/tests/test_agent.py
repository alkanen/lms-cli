"""Tests for ai_cli.core.agent data structures."""

from ai_cli.core.agent import AgentResult, AgentSpec, BackendConfig


class TestBackendConfig:
    def test_defaults(self):
        bc = BackendConfig(base_url="http://localhost:11434/v1")
        assert bc.base_url == "http://localhost:11434/v1"
        assert bc.api_key_env is None

    def test_custom_api_key_env(self):
        bc = BackendConfig(base_url="http://example.com", api_key_env="MY_API_KEY")
        assert bc.api_key_env == "MY_API_KEY"


class TestAgentSpec:
    def test_required_fields(self):
        spec = AgentSpec(
            name="test",
            system_message="You are a test agent.",
            tools=["read_file"],
            model="llama3.2:3b",
        )
        assert spec.name == "test"
        assert spec.system_message == "You are a test agent."
        assert spec.tools == ["read_file"]
        assert spec.model == "llama3.2:3b"

    def test_defaults(self):
        spec = AgentSpec(name="t", system_message="m", tools=[], model="m")
        assert spec.max_response_tokens == 4096
        assert spec.persistence == "ephemeral"
        assert spec.backend is None
        assert spec.tool_permission_overrides == {}
        assert spec.max_tool_rounds == 10
        assert spec.context_limit_threshold == 0.90

    def test_custom_values(self):
        backend = BackendConfig(
            base_url="http://localhost:11435/v1", api_key_env="OLLAMA_KEY"
        )
        spec = AgentSpec(
            name="coder",
            system_message="Write code.",
            tools=["read_file", "write_file"],
            model="qwen2.5-coder:14b",
            max_response_tokens=8192,
            persistence="session",
            backend=backend,
            tool_permission_overrides={"write_file": False},
            max_tool_rounds=20,
            context_limit_threshold=0.85,
        )
        assert spec.max_response_tokens == 8192
        assert spec.persistence == "session"
        assert spec.backend is backend
        assert spec.tool_permission_overrides == {"write_file": False}
        assert spec.max_tool_rounds == 20
        assert spec.context_limit_threshold == 0.85

    def test_tool_permission_overrides_independent_instances(self):
        """Default dict should not be shared between instances."""
        a = AgentSpec(name="a", system_message="m", tools=[], model="m")
        b = AgentSpec(name="b", system_message="m", tools=[], model="m")
        a.tool_permission_overrides["x"] = True
        assert "x" not in b.tool_permission_overrides


class TestAgentResult:
    def test_ok_result(self):
        r = AgentResult(text="Done.", status="ok")
        assert r.text == "Done."
        assert r.status == "ok"
        assert r.partial is False
        assert r.error_message == ""

    def test_context_limit(self):
        r = AgentResult(text="Partial output.", status="context_limit", partial=True)
        assert r.partial is True

    def test_error_result(self):
        r = AgentResult(
            text="",
            status="error",
            partial=True,
            error_message="Connection refused",
        )
        assert r.status == "error"
        assert r.error_message == "Connection refused"

    def test_tool_limit(self):
        r = AgentResult(text="halfway", status="tool_limit", partial=True)
        assert r.status == "tool_limit"
        assert r.partial is True
