"""Run one synthesized Python script in a locked-down container."""

import subprocess
from pathlib import Path

from aqe.config import EngineConfig
from aqe.errors import HarnessError


class DockerSandbox:
    """One container per step. No privileged mode and no Docker socket."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def run(self, script: str, evidence_dir: Path, *, network: bool = False) -> tuple[str, str]:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        work = evidence_dir.parent / "work"
        work.mkdir(parents=True, exist_ok=True)
        script_path = work / "script.py"
        script_path.write_text(script, encoding="utf-8")
        
        # Mount the run directory so CODING step files are accessible
        # CODING files are created in evidence_dir.parent (the run directory)
        run_dir = evidence_dir.parent
        
        command = ["docker", "run", "--rm"]
        if not network:
            command.extend(["--network", "none"])
        command.extend([
            "--read-only",
            "--tmpfs",
            "/tmp:rw,size=32m",
            "--memory",
            "256m",
            "--cpus",
            "0.5",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "10001:10001",
            "-v",
            f"{evidence_dir.resolve()}:/evidence:ro",
            "-v",
            f"{run_dir.resolve()}:/run:ro",
            "-v",
            f"{script_path.resolve()}:/opt/script.py:ro",
            self.config.sandbox_image,
            "python",
            "/opt/script.py",
        ])
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.config.sandbox_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise HarnessError("driver_timeout", "The sandbox timed out.") from exc
        except OSError as exc:
            raise HarnessError("sandbox_start_failed", f"Docker sandbox failed to start: {exc}") from exc
        stdout = (completed.stdout or "")[: self.config.sandbox_output_limit]
        stderr = (completed.stderr or "")[: self.config.sandbox_output_limit]
        if completed.returncode != 0 and _container_did_not_start(completed.returncode, stderr):
            raise HarnessError(
                "sandbox_start_failed",
                f"Docker sandbox failed to start: {stderr or stdout}",
            )
        return stdout, stderr


def _container_did_not_start(returncode: int, stderr: str) -> bool:
    if returncode in {125, 126, 127}:
        return True
    lowered = stderr.lower()
    markers = (
        "cannot connect",
        "unable to find image",
        "error during connect",
        "failed to start",
        "no such image",
    )
    return any(marker in lowered for marker in markers)


def build_sandbox(config: EngineConfig) -> DockerSandbox:
    return DockerSandbox(config)
