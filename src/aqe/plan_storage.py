"""Plan storage and persistence using file system."""

import json
import logging
from pathlib import Path

from aqe.state import PlanStorage, PlannerChatHistory

logger = logging.getLogger(__name__)


class PlanStorageManager:
    """Manages plan storage in the file system."""

    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir

    def _get_plan_file(self, run_id: str) -> Path:
        """Get the plan file path for a run."""
        return self.runs_dir / run_id / "plan.json"

    def _get_chat_file(self, run_id: str) -> Path:
        """Get the chat history file path for a run."""
        return self.runs_dir / run_id / "planner_chat.json"

    def save_plan(self, plan: PlanStorage) -> None:
        """Save plan to file system."""
        plan_file = self._get_plan_file(plan.run_id)
        plan_file.parent.mkdir(parents=True, exist_ok=True)
        with open(plan_file, "w") as f:
            json.dump(plan.model_dump(mode="json"), f, indent=2, default=str)
        logger.info(f"Saved plan for run {plan.run_id}")

    def load_plan(self, run_id: str) -> PlanStorage | None:
        """Load plan from file system."""
        plan_file = self._get_plan_file(run_id)
        if not plan_file.exists():
            return None
        with open(plan_file, "r") as f:
            data = json.load(f)
        return PlanStorage.model_validate(data)

    def save_chat_history(self, chat: PlannerChatHistory) -> None:
        """Save chat history to file system."""
        chat_file = self._get_chat_file(chat.run_id)
        chat_file.parent.mkdir(parents=True, exist_ok=True)
        with open(chat_file, "w") as f:
            json.dump(chat.model_dump(mode="json"), f, indent=2, default=str)
        logger.info(f"Saved chat history for run {chat.run_id}")

    def load_chat_history(self, run_id: str) -> PlannerChatHistory | None:
        """Load chat history from file system."""
        chat_file = self._get_chat_file(run_id)
        if not chat_file.exists():
            return None
        with open(chat_file, "r") as f:
            data = json.load(f)
        return PlannerChatHistory.model_validate(data)

    def delete_plan(self, run_id: str) -> None:
        """Delete plan file."""
        plan_file = self._get_plan_file(run_id)
        if plan_file.exists():
            plan_file.unlink()
            logger.info(f"Deleted plan for run {run_id}")

    def delete_chat_history(self, run_id: str) -> None:
        """Delete chat history file."""
        chat_file = self._get_chat_file(run_id)
        if chat_file.exists():
            chat_file.unlink()
            logger.info(f"Deleted chat history for run {run_id}")
