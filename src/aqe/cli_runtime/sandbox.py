"""Run one synthesized Python script in a locked-down container."""

import logging
import subprocess
import uuid
from pathlib import Path

from aqe.config import EngineConfig
from aqe.errors import HarnessError

logger = logging.getLogger(__name__)


def determine_execution_context(command: str, default_image: str) -> tuple[str, bool]:
    """Use LLM to determine the appropriate Docker image and whether to run on host.
    
    Returns:
        tuple: (docker_image, run_on_host)
            - docker_image: The Docker image to use (or default)
            - run_on_host: True if command should run on host, False if in sandbox
    """
    import json
    
    try:
        from aqe.chat import get_chat_model
        chat_model = get_chat_model()
        
        prompt = f"""Given this command: {command}

Analyze this command and determine:
1. What Docker image would be appropriate to run this command?
2. Should this command run on the host machine instead of in a Docker container?

Consider:
- If the command uses Docker CLI (docker build, docker run, etc.), it CANNOT run in a container → run_on_host=true, image={default_image}
- If the command needs host system access (systemctl, host networking, etc.) → run_on_host=true, image={default_image}
- If the command uses Python, suggest a Python image (e.g., python:3.12-slim) → run_on_host=false
- If the command uses Node.js, suggest a Node image (e.g., node:20-slim) → run_on_host=false
- If the command uses curl/wget, suggest a minimal image with curl → run_on_host=false
- If unsure, use default image and run in sandbox → run_on_host=false, image={default_image}

Return your answer in this exact JSON format:
{{"image": "image_name", "run_on_host": true/false}}

Example responses:
- For "docker build -t test": {{"image": "aqe-sandbox:local", "run_on_host": true}}
- For "python script.py": {{"image": "python:3.12-slim", "run_on_host": false}}
- For "npm install": {{"image": "node:20-slim", "run_on_host": false}}"""

        # Use LangChain's invoke method instead of generate
        from langchain_core.messages import HumanMessage
        response = chat_model.invoke([HumanMessage(content=prompt)])
        
        # Extract content from AIMessage
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON response
        result = json.loads(response_text.strip())
        
        image = result.get("image", default_image)
        run_on_host = result.get("run_on_host", False)
        
        # Validate the image name
        if not image or len(image) == 0:
            image = default_image
        
        return image, run_on_host
    except (ImportError, AttributeError, ValueError, json.JSONDecodeError, KeyError) as exc:
        logger.warning(f"Failed to determine execution context with LLM: {exc}")
    
    return default_image, False


class DockerSandbox:
    """Shared container per run. No privileged mode and no Docker socket."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._container_id: str | None = None
        self._container_name: str | None = None
        self._current_image: str | None = None

    def start_container(self, run_dir: Path, work_dir: Path, *, network: bool = True, image: str | None = None) -> str:
        """Start a persistent container for the run."""
        if self._container_id is not None:
            # If image changed, stop and restart with new image
            if image and image != self._current_image:
                self.stop_container()
            else:
                return self._container_id

        # Generate a unique container name that is available
        max_attempts = 10
        for attempt in range(max_attempts):
            self._container_name = f"aqe-sandbox-{uuid.uuid4().hex[:12]}"
            
            # Check if container name is already in use
            try:
                result = subprocess.run(
                    ["docker", "inspect", "--format='{{.Id}}'", self._container_name],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                # If container exists, remove it
                if result.returncode == 0 and result.stdout.strip():
                    logger.info(f"Container {self._container_name} already exists, removing it")
                    subprocess.run(
                        ["docker", "rm", "-f", self._container_name],
                        capture_output=True,
                        timeout=10,
                        check=False,
                    )
                    break  # Name is now available
                else:
                    break  # Name is available
            except (subprocess.TimeoutExpired, OSError) as exc:
                logger.warning(f"Failed to check container name availability: {exc}")
                if attempt == max_attempts - 1:
                    raise HarnessError("container_name_check_failed", f"Failed to ensure container name is available after {max_attempts} attempts")
                continue  # Try with a new name
        
        run_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        # Use provided image or default
        image_to_use = image or self.config.sandbox_image
        self._current_image = image_to_use

        command = ["docker", "run", "-d"]
        if not network:
            command.extend(["--network", "none"])
        command.extend([
            "--name", self._container_name,
            "--read-only",
            "--tmpfs", "/tmp:rw,size=2048m",
            "--memory", "256m",
            "--cpus", "0.5",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "-e", "PIP_CACHE_DIR=/tmp/.cache/pip",
            "-e", "PIP_NO_CACHE_DIR=false",
            "-e", "XDG_CACHE_HOME=/tmp/.cache",
            "-e", "PYTHONUSERBASE=/tmp/.local",
            "-e", "PATH=/tmp/.local/bin:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "-v", f"{run_dir.resolve()}:/run:rw",
            "-v", f"{work_dir.resolve()}:/work:rw",
            image_to_use,
            "tail", "-f", "/dev/null",  # Keep container running
        ])

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            self._container_id = result.stdout.strip()
            return self._container_id
        except subprocess.TimeoutExpired as exc:
            raise HarnessError("driver_timeout", "Failed to start sandbox container.") from exc
        except subprocess.CalledProcessError as exc:
            raise HarnessError(
                "sandbox_start_failed",
                f"Failed to start sandbox container: {exc.stderr}",
            ) from exc
        except OSError as exc:
            raise HarnessError("sandbox_start_failed", f"Docker sandbox failed to start: {exc}") from exc

    def stop_container(self) -> None:
        """Stop and remove the persistent container."""
        if self._container_id is None:
            return

        try:
            subprocess.run(
                ["docker", "stop", self._container_id],
                capture_output=True,
                timeout=10,
                check=False,
            )
            subprocess.run(
                ["docker", "rm", self._container_id],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            pass  # Best effort cleanup
        finally:
            self._container_id = None
            self._container_name = None

    def run(self, script: str, evidence_dir: Path, work_dir: Path, *, network: bool = False) -> tuple[str, str]:
        """Run a script or command in the persistent container."""
        if self._container_id is None:
            raise HarnessError("sandbox_not_started", "Sandbox container not started. Call start_container first.")

        evidence_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        # Check if this is a direct CLI command (not a Python script)
        # Direct CLI commands are simple commands like "curl http://example.com"
        is_direct_cli = (
            not script.strip().startswith(("import", "from", "def", "class", "#", '"', "'")) and
            not any(keyword in script for keyword in ["import ", "from ", "def ", "class ", "print("])
        )

        if is_direct_cli:
            # Execute as direct CLI command using bash
            command = [
                "docker", "exec",
                self._container_id,
                "sh", "-c", f"cd /work && {script}",
            ]
        else:
            # Determine the appropriate interpreter based on the image
            interpreter = "python"
            if self._current_image:
                if self._current_image.startswith("node:"):
                    interpreter = "node"
                elif self._current_image.startswith("alpine:") or self._current_image.startswith("curlimages/"):
                    interpreter = "sh"
            
            # Execute as script with appropriate interpreter
            script_path = work_dir / "script.py"
            script_path.write_text(script, encoding="utf-8")
            command = [
                "docker", "exec",
                self._container_id,
                interpreter, "/work/script.py",
            ]

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
            raise HarnessError("sandbox_start_failed", f"Docker sandbox failed to execute: {exc}") from exc

        stdout = (completed.stdout or "")[: self.config.sandbox_output_limit]
        stderr = (completed.stderr or "")[: self.config.sandbox_output_limit]
        if completed.returncode != 0 and _container_did_not_start(completed.returncode, stderr):
            raise HarnessError(
                "sandbox_start_failed",
                f"Docker sandbox failed to execute: {stderr or stdout}",
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
