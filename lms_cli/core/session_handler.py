"""
Session handler for managing message history and compaction.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from slugify import slugify


SESSIONS_DIR = Path.home() / ".lms-cli" / "sessions"


class SessionHandler:
    """
    Handles session storage and retrieval for message histories.

    Attributes:
        session_id: A unique identifier for the session.
        session_dir: Directory where session files are stored.
        complete_history_path: Path to the complete history file.
        recent_history_path: Path to the recent history file.
    """

    def __init__(self, workspace_path: str, sessions_dir: Optional[str] = None):
        """
        Initialize the SessionHandler with a workspace path.

        Args:
            workspace_path: The path to the workspace directory.
            sessions_dir: Optional path to the directory where session files are stored.
                If not provided, the default SESSIONS_DIR is used.
        """
        self.session_id = self._generate_session_id(str(Path(workspace_path).resolve()))
        self.session_dir = Path(sessions_dir if sessions_dir else SESSIONS_DIR)
        self.complete_history_path = self.session_dir / f"{self.session_id}_complete.jsonl"
        self.recent_history_path = self.session_dir / f"{self.session_id}_recent.jsonl"

    def _generate_session_id(self, workspace_path: str) -> str:
        """
        Generate a session ID based on the workspace path and current time.

        Args:
            workspace_path: The path to the workspace directory.

        Returns:
            A slugified session ID.
        """
        timestamp = datetime.strftime(datetime.now(), "%Y-%m-%dT%Hh%Mm%Ss")
        return f"{slugify(workspace_path)}_{timestamp}"

    def _ensure_session_dir(self):
        """
        Ensure the session directory exists.
        """
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def save_message(
        self,
        message: Dict[str, str],
        append: bool = True,
    ):
        """
        Save a message to both complete and recent history files.
        Args:
            message: The message to save.
            append: Whether to append the message or overwrite the file.
        """
        self.save_message_to_complete_history(message, append)
        self.save_message_to_recent_history(message, append)

    def save_message_to_complete_history(
        self,
        message: Dict[str, str],
        append: bool = True,
    ):
        """
        Save a message to the complete history file.

        Args:
            message: The message to save.
            append: Whether to append the message or overwrite the file.
        """
        self._ensure_session_dir()
        mode = "a" if append else "w"
        with open(self.complete_history_path, mode, encoding="utf-8") as f:
            json.dump(message, f)
            f.write("\n")

    def save_message_to_recent_history(
        self,
        message: Dict[str, str],
        append: bool = True,
    ):
        """
        Save a message to the recent history file.

        Args:
            message: The message to save.
            append: Whether to append the message or overwrite the file.
        """
        self._ensure_session_dir()
        mode = "a" if append else "w"
        with open(self.recent_history_path, mode, encoding="utf-8") as f:
            json.dump(message, f)
            f.write("\n")

    def load_complete_history(self) -> List[Dict[str, str]]:
        """
        Load the complete message history.

        Returns:
            A list of messages in the complete history.
        """
        if not self.complete_history_path.exists():
            return []

        with open(self.complete_history_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def load_recent_history(self) -> List[Dict[str, str]]:
        """
        Load the recent message history.

        Returns:
            A list of messages in the recent history.
        """
        if not self.recent_history_path.exists():
            return []

        with open(self.recent_history_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def compact_recent_history(self) -> List[Dict[str, str]]:
        """
        Compact the recent history by removing everything but the initial system
        message and the last two regular messages on the assumption that they
        are a request for the AI to summarize everything followed by the agent's
        summary response.

        Returns:
            A list of the compacted history
        """
        # Load the complete and recent histories
        complete_history = self.load_complete_history()
        recent_history = self.load_recent_history()

        # Nothing to compact
        if len(complete_history) <= 3:
            return complete_history

        # Overwrite the recent history with the system message and summary
        compacted_history: List[Dict[str, str]] = [complete_history[0]]
        compacted_history.extend(recent_history[-2:])  # Keep compaction request and response

        # Write the compacted history to the recent file
        with open(self.recent_history_path, "w", encoding="utf-8") as f:
            for message in compacted_history:
                json.dump(message, f)
                f.write("\n")

        return compacted_history

    @classmethod
    def list_available_sessions(cls, sessions_dir: Optional[str] = None) -> List[str]:
        """
        List all available sessions in the session directory.

        Args:
            sessions_dir: Optional custom directory for storing sessions. Defaults to SESSIONS_DIR.

        Returns:
            A list of session IDs (filenames without extension).
        """
        session_dir = Path(sessions_dir if sessions_dir else SESSIONS_DIR)
        if not session_dir.exists():
            return []

        sessions = set()
        for file in session_dir.iterdir():
            if file.is_file() and file.suffix == ".jsonl":
                # Extract the session ID from the filename (e.g., "session_id_complete.jsonl" -> "session_id")
                session_id = file.stem.rsplit("_", 1)[0]
                sessions.add(session_id)

        return sorted(sessions, reverse=True)

    @classmethod
    def restore_session(cls, session_id: str, sessions_dir: Optional[str] = None) -> Optional["SessionHandler"]:
        """
        Restore a session by loading its complete and recent histories into a new SessionHandler instance.

        Args:
            session_id: The ID of the session to restore.
            sessions_dir: Optional custom directory for storing sessions. Defaults to SESSIONS_DIR.

        Returns:
            A SessionHandler object with the restored history, or None if the session does not exist.
        """
        # Check if the session exists
        available_sessions = cls.list_available_sessions(sessions_dir)
        if session_id not in available_sessions:
            return None

        # Create a new SessionHandler instance with the restored session ID
        session_dir = Path(sessions_dir) if sessions_dir else SESSIONS_DIR
        restored_handler = cls.__new__(cls)  # Create an instance without calling __init__
        restored_handler.session_id = session_id
        restored_handler.session_dir = session_dir
        restored_handler.complete_history_path = session_dir / f"{session_id}_complete.jsonl"
        restored_handler.recent_history_path = session_dir / f"{session_id}_recent.jsonl"

        return restored_handler
