"""Pi coding agent subprocess management."""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from aqe.errors import HarnessError
from aqe.state import ActionResult, CodingAction, TestStep


class PiAgentError(HarnessError):
    """Pi agent specific errors."""

    def __init__(self, code: str, message: str):
        super().__init__(code, message)


class CodingSubsystem:
    """Manages Pi coding agent subprocess for code operations."""

    def __init__(self, work_dir: Path | None = None, pi_llm_model: str | None = None) -> None:
        self.work_dir = work_dir or Path(tempfile.gettempdir())
        self._process: subprocess.Popen[bytes] | None = None
        self._rpc_port = 8766  # Default RPC port for Pi agent
        self.pi_llm_model = pi_llm_model

    def _check_pi_available(self) -> None:
        """Check if Pi agent is installed and available."""
        if shutil.which("pi") is None:
            raise PiAgentError(
                "coding_agent_not_available",
                "Pi coding agent is not installed. Please install Pi to use coding operations.",
            )

    def _start_pi_agent(self, run_id: str) -> subprocess.Popen[bytes]:
        """Start Pi agent in RPC mode."""
        run_dir = self.work_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        work_dir = run_dir / "work"
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            cmd = ["pi", "--mode", "rpc"]
            if self.pi_llm_model:
                cmd.extend(["--model", self.pi_llm_model])

            process = subprocess.Popen(
                cmd,
                cwd=str(work_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
            )
            return process
        except (OSError, subprocess.SubprocessError) as exc:
            raise PiAgentError(
                "coding_agent_failed",
                f"Failed to start Pi agent subprocess: {exc}",
            ) from exc

    def _stop_pi_agent(self) -> None:
        """Stop the Pi agent subprocess if running."""
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    self._process.kill()
                    self._process.wait(timeout=2)
                except (subprocess.TimeoutExpired, OSError):
                    pass
            finally:
                self._process = None

    def _execute_coding_action(self, action: CodingAction, run_id: str) -> dict[str, Any]:
        """Execute a single coding action through Pi agent."""
        # For simplified implementation, we don't need Pi agent
        # Skip Pi availability check and subprocess management
        # In full implementation, this would use RPC communication with Pi

        try:
            # Use simplified subprocess execution
            return self._execute_via_subprocess(action, run_id)
        except (OSError, subprocess.SubprocessError) as exc:
            raise PiAgentError(
                "coding_agent_failed",
                f"Pi agent execution failed: {exc}",
            ) from exc

    def _prepare_rpc_request(self, action: CodingAction) -> dict[str, Any]:
        """Prepare RPC request for Pi agent."""
        # This would be used for actual RPC communication
        return {
            "action": action.action,
            "file_path": action.file_path,
            "content": action.content,
            "description": action.description,
        }

    def _work_file(self, work_dir: Path, file_path: str) -> Path:
        relative = Path(file_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise PiAgentError(
                "coding_agent_failed",
                f"File path is outside the work directory: {file_path}",
            )
        target = (work_dir / relative).resolve()
        root = work_dir.resolve()
        if target != root and root not in target.parents:
            raise PiAgentError(
                "coding_agent_failed",
                f"File path is outside the work directory: {file_path}",
            )
        return target

    def _written(self, action: CodingAction, target: Path, message: str) -> dict[str, Any]:
        content = target.read_text(encoding="utf-8")
        return {
            "success": True,
            "action": action.action,
            "message": message,
            "file_path": action.file_path,
            "content": content,
        }

    def _execute_via_subprocess(self, action: CodingAction, run_id: str) -> dict[str, Any]:
        """Execute coding action via subprocess (simplified implementation)."""
        # Use work_dir directly as the base directory for files
        # run_id is now the actual run ID (e.g., "8355f89ae0fc4dcba9ef10da1c4b1f5d")
        run_dir = self.work_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        work_dir = run_dir / "work"
        work_dir.mkdir(parents=True, exist_ok=True)

        if action.action == "create_file":
            if not action.file_path or action.content is None:
                raise PiAgentError(
                    "coding_agent_failed",
                    "create_file action requires file_path and content",
                )
            file_path = self._work_file(work_dir, action.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(action.content, encoding="utf-8")
            return self._written(action, file_path, f"Created file: {action.file_path}")

        elif action.action == "update_file":
            if not action.file_path:
                raise PiAgentError(
                    "coding_agent_failed",
                    "update_file action requires file_path",
                )
            file_path = self._work_file(work_dir, action.file_path)
            if not file_path.exists():
                raise PiAgentError(
                    "coding_agent_failed",
                    f"File not found: {action.file_path}",
                )
            if action.content is not None:
                file_path.write_text(action.content, encoding="utf-8")
            return self._written(action, file_path, f"Updated file: {action.file_path}")

        elif action.action == "review_code":
            if not action.file_path:
                raise PiAgentError(
                    "coding_agent_failed",
                    "review_code action requires file_path",
                )
            file_path = self._work_file(work_dir, action.file_path)
            if not file_path.exists():
                raise PiAgentError(
                    "coding_agent_failed",
                    f"File not found for review: {action.file_path}",
                )
            return self._written(action, file_path, f"Reviewed file: {action.file_path}")

        elif action.action == "execute_code":
            # Code execution is now handled by CLI steps, not CODING steps
            raise PiAgentError(
                "coding_agent_failed",
                "execute_code action should be handled by CLI steps, not CODING steps",
            )

        else:
            raise PiAgentError(
                "coding_agent_failed",
                f"Unknown coding action: {action.action}",
            )

    def execute_coding_action(self, step: TestStep, evidence_dir: Path) -> ActionResult:
        """Execute coding operations for a test step."""
        if not step.coding_operations:
            return ActionResult(
                ok=False,
                summary="No coding operations to execute",
                evidence={"error": "No coding operations defined for this step"},
            )

        results = []
        errors = []

        # Use evidence_dir parent name as run_id for isolation
        run_id = evidence_dir.parent.name

        for operation in step.coding_operations:
            try:
                result = self._execute_coding_action(operation, run_id)
                results.append(result)
            except PiAgentError as exc:
                errors.append(str(exc))
                results.append({
                    "success": False,
                    "action": operation.action,
                    "file_path": operation.file_path,
                    "error": str(exc),
                })
            except Exception as exc:  # noqa: BLE001
                # Catch any unexpected errors to prevent engine_error
                errors.append(f"Unexpected error: {exc}")
                results.append({
                    "success": False,
                    "action": operation.action,
                    "file_path": operation.file_path,
                    "error": str(exc),
                })

        # Note: Pi agent cleanup not needed for simplified implementation
        # self._stop_pi_agent()

        if errors:
            return ActionResult(
                ok=False,
                summary=f"Coding operations failed: {'; '.join(errors)}",
                evidence={
                    "results": results,
                    "errors": errors,
                    "operation_count": len(step.coding_operations),
                },
            )

        summary = f"Executed {len(results)} coding operation(s) successfully"
        return ActionResult(
            ok=True,
            summary=summary,
            evidence={
                "results": results,
                "operation_count": len(step.coding_operations),
                "summary": summary,
            },
        )

    def close(self) -> None:
        """Clean up resources."""
        self._stop_pi_agent()


def build_coding_agent(work_dir: Path | None = None, pi_llm_model: str | None = None) -> CodingSubsystem:
    """Factory function to create a coding subsystem."""
    if work_dir is None:
        work_dir = Path(tempfile.gettempdir())
    return CodingSubsystem(work_dir, pi_llm_model)
