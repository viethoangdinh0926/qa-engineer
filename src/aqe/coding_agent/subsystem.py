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

        try:
            cmd = ["pi", "--mode", "rpc"]
            if self.pi_llm_model:
                cmd.extend(["--model", self.pi_llm_model])

            process = subprocess.Popen(
                cmd,
                cwd=str(run_dir),
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
        self._check_pi_available()

        # Start Pi agent if not already running
        if self._process is None or self._process.poll() is not None:
            self._process = self._start_pi_agent(run_id)

        # Prepare the RPC request based on action type
        request = self._prepare_rpc_request(action)

        try:
            # For now, we'll use a simpler approach: execute directly via subprocess
            # In a full implementation, this would use RPC communication
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

    def _execute_via_subprocess(self, action: CodingAction, run_id: str) -> dict[str, Any]:
        """Execute coding action via subprocess (simplified implementation)."""
        run_dir = self.work_dir / run_id

        if action.action == "create_file":
            if not action.file_path or not action.content:
                raise PiAgentError(
                    "coding_agent_failed",
                    "create_file action requires file_path and content",
                )
            file_path = run_dir / action.file_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(action.content, encoding="utf-8")
            return {
                "success": True,
                "message": f"Created file: {action.file_path}",
                "file_path": str(file_path),
            }

        elif action.action == "update_file":
            if not action.file_path:
                raise PiAgentError(
                    "coding_agent_failed",
                    "update_file action requires file_path",
                )
            file_path = run_dir / action.file_path
            if not file_path.exists():
                raise PiAgentError(
                    "coding_agent_failed",
                    f"File not found: {action.file_path}",
                )
            if action.content:
                file_path.write_text(action.content, encoding="utf-8")
            return {
                "success": True,
                "message": f"Updated file: {action.file_path}",
                "file_path": str(file_path),
            }

        elif action.action == "review_code":
            if not action.file_path:
                raise PiAgentError(
                    "coding_agent_failed",
                    "review_code action requires file_path",
                )
            file_path = run_dir / action.file_path
            if not file_path.exists():
                raise PiAgentError(
                    "coding_agent_failed",
                    f"File not found for review: {action.file_path}",
                )
            content = file_path.read_text(encoding="utf-8")
            return {
                "success": True,
                "message": f"Reviewed file: {action.file_path}",
                "file_path": str(file_path),
                "content": content,
                "line_count": len(content.splitlines()),
            }

        elif action.action == "execute_code":
            # For code execution, we might use Pi agent or direct execution
            if not action.file_path:
                raise PiAgentError(
                    "coding_agent_failed",
                    "execute_code action requires file_path",
                )
            file_path = run_dir / action.file_path
            if not file_path.exists():
                raise PiAgentError(
                    "coding_agent_failed",
                    f"File not found for execution: {action.file_path}",
                )
            try:
                result = subprocess.run(
                    ["python", str(file_path)],
                    cwd=str(run_dir),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return {
                    "success": result.returncode == 0,
                    "message": f"Executed file: {action.file_path}",
                    "file_path": str(file_path),
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                }
            except subprocess.TimeoutExpired as exc:
                raise PiAgentError(
                    "coding_agent_failed",
                    f"Code execution timed out: {exc}",
                ) from exc
            except (OSError, subprocess.SubprocessError) as exc:
                raise PiAgentError(
                    "coding_agent_failed",
                    f"Code execution failed: {exc}",
                ) from exc

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

        for operation in step.coding_operations:
            try:
                result = self._execute_coding_action(operation, evidence_dir.name)
                results.append(result)
            except PiAgentError as exc:
                errors.append(str(exc))
                results.append({
                    "success": False,
                    "error": str(exc),
                    "action": operation.action,
                })

        # Stop Pi agent after execution
        self._stop_pi_agent()

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
