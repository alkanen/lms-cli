"""
Context manager for the CLI tool.
Manages configuration and core component instances.
"""

from typing import Callable, List, Tuple
import yaml

from lms_cli.core.embedding_manager import EmbeddingManager
from lms_cli.core.lm_studio_client import LMStudioClient
from lms_cli.core.session_handler import SessionHandler
from lms_cli.core.tool_registry import ToolRegistry
from lms_cli.core.workspace import Workspace


class CLIContext:
    """
    Context manager for the CLI tool.

    Attributes:
        config_path: Path to the configuration file.
        workspace_root: Root path of the workspace.
        workspace: Instance of Workspace.
        lm_studio_client: Instance of LMStudioClient.
        embedding_manager: Instance of EmbeddingManager.
        session_handler: Instance of SessionHandler.
        tool_registry: Instance of ToolRegistry.
        config: Loaded configuration dictionary.
        permission_callback: Called by tools needing to ask for permission.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        workspace_root: str = ".",
        model: str | None = None,
        base_url: str | None = None,
        permission_callback: Callable[[str, List[str]], Tuple[int, str]] | None = None,
    ):
        """
        Initialize the CLI context.

        Args:
            config_path: Path to the configuration file.
            workspace_root: Root path of the workspace.
            model: Optional model name for LMStudioClient.
            base_url: Optional base URL for LMStudioClient.
            permission_callback: Optional permission callback function.
        """
        self.config_path = config_path
        self.workspace_root = workspace_root
        self.permission_callback = permission_callback

        # Configuration
        self.config = self._load_config()
        if not model:
            model = self.config["lm_studio"]["model"]
        if not base_url:
            base_url = self.config["lm_studio"]["base_url"]

        # Initialize core components
        self.workspace = Workspace(workspace_root)
        self.lm_studio_client = LMStudioClient(self.config_path, model, base_url)
        self.embedding_manager = EmbeddingManager(self.config_path)
        self.session_handler = SessionHandler(self.workspace)
        self.tool_registry = ToolRegistry(self)

    def _load_config(self) -> dict:
        """
        Load configuration from the config file.

        Returns:
            dict: Loaded configuration dictionary.
        """
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def __enter__(self):
        """
        Enter the runtime context and return the context instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context.
        """
        pass
