"""Ask the planner for a Python snippet and run it in the sandbox."""

import logging
import os
import re
import shlex
import subprocess
from pathlib import Path

from aqe.cli_runtime.sandbox import DockerSandbox, determine_execution_context
from aqe.llm import Planner, command_from_intent, command_needs_network
from aqe.state import ActionResult, TestStep

logger = logging.getLogger(__name__)


def ensure_container_name_available(container_name: str) -> bool:
    """Check if a container name is available and remove existing container if needed."""
    try:
        # Check if container name is already in use
        result = subprocess.run(
            ["docker", "inspect", "--format='{{.Id}}'", container_name],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        # If container exists, remove it
        if result.returncode == 0 and result.stdout.strip():
            logger.info(f"Container {container_name} already exists, removing it")
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                timeout=10,
                check=False,
            )
        return True
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning(f"Failed to check container name availability for {container_name}: {exc}")
        return False


class CLISubsystem:
    def __init__(self, planner: Planner, sandbox: DockerSandbox) -> None:
        self.planner = planner
        self.sandbox = sandbox

    def _is_service_setup_command(self, command: str) -> bool:
        """Detect if a command is for service setup rather than test execution."""
        service_patterns = [
            "nohup",
            "start service",
            "start server",
            "background",
            "daemon",
            "&",  # Background operator
            "service start",
            "systemctl start",
            "pip install",  # Package installation
            "npm install",  # Node package installation
            "yarn install",  # Yarn package installation
            "docker",  # Docker commands (must run on host)
        ]
        command_lower = command.lower()
        return any(pattern in command_lower for pattern in service_patterns)

    def _execute_on_host(self, command: str, work_dir: Path) -> tuple[str, str]:
        """Execute a command on the host machine (outside sandbox)."""
        try:
            # Clean the command string - remove any problematic characters
            # Remove any null bytes or other control characters
            command = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', command)

            # Check for container names in Docker commands and ensure they're available
            if "docker" in command.lower():
                # Look for --name flag in Docker commands
                name_match = re.search(r'--name\s+(\S+)', command)
                if name_match:
                    container_name = name_match.group(1)
                    # Ensure the container name is available
                    if not ensure_container_name_available(container_name):
                        logger.warning(f"Could not ensure container name {container_name} is available, proceeding anyway")

            # Set up environment variables for consistent Python package location
            env = {
                "PYTHONUSERBASE": str(work_dir / ".local"),
                "PATH": f"{work_dir / '.local' / 'bin'}:{os.environ.get('PATH', '')}",
                "PIP_CACHE_DIR": str(work_dir / ".cache" / "pip"),
                "XDG_CACHE_HOME": str(work_dir / ".cache"),
            }
            
            # Handle multi-line commands by splitting them and executing sequentially
            commands = [cmd.strip() for cmd in command.split('\n') if cmd.strip()]
            if not commands:
                return "", "No commands to execute"

            all_stdout = []
            all_stderr = []

            # Check if any command has shell variables that need to be shared across lines
            has_shared_variables = any('$' in cmd for cmd in commands)
            
            if has_shared_variables:
                # Execute all commands in a single shell session to preserve variable scope
                combined_command = "\n".join(commands)
                logger.info(f"Executing combined host command with shared variables: {combined_command!r}")
                
                result = subprocess.run(
                    combined_command,
                    shell=True,
                    executable="/bin/bash",
                    cwd=str(work_dir),
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                    env={**os.environ, **env},
                )
                all_stdout.append(result.stdout)
                all_stderr.append(result.stderr)
            else:
                # Execute commands separately for better error isolation
                for cmd in commands:
                    # Debug: log the command being executed
                    logger.info(f"Executing host command: {cmd!r}")

                    # Check if command contains shell syntax that requires shell execution
                    has_shell_syntax = any(char in cmd for char in ['$', '`', '|', '&&', '||', ';', '>', '<', '*'])

                    if has_shell_syntax:
                        # Use shell execution for commands with shell syntax
                        result = subprocess.run(
                            cmd,
                            shell=True,
                            executable="/bin/bash",
                            cwd=str(work_dir),
                            capture_output=True,
                            text=True,
                            timeout=120,
                            check=False,
                            env={**os.environ, **env},
                        )
                        all_stdout.append(result.stdout)
                        all_stderr.append(result.stderr)
                    else:
                        # Try array-based execution for simple commands
                        try:
                            cmd_parts = shlex.split(cmd)
                            result = subprocess.run(
                                cmd_parts,
                                cwd=str(work_dir),
                                capture_output=True,
                                text=True,
                                timeout=120,
                                check=False,
                                env={**os.environ, **env},
                            )
                            all_stdout.append(result.stdout)
                            all_stderr.append(result.stderr)
                        except ValueError:
                            # Fallback to shell execution if splitting fails
                            result = subprocess.run(
                                cmd,
                                shell=True,
                                executable="/bin/bash",
                                cwd=str(work_dir),
                                capture_output=True,
                                text=True,
                                timeout=120,
                                check=False,
                                env={**os.environ, **env},
                            )
                            all_stdout.append(result.stdout)
                            all_stderr.append(result.stderr)

            stdout = "\n".join(all_stdout)
            stderr = "\n".join(all_stderr)
            return stdout, stderr
        except subprocess.TimeoutExpired:
            return "", "Command timed out on host machine"
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            logger.error(f"Host execution error: {e!s}, command was: {command!r}")
            return "", f"Host execution failed: {e!s}"

    def execute_runtime_action(self, step: TestStep, evidence_dir: Path) -> ActionResult:
        checks = step.verifications or [step.assertion]
        intent = step.action + "\n" + "\n".join(f"Assertion: {item}" for item in checks)
        script = self.planner.script_for(intent)
        command = command_from_intent(intent) or ""
        work_dir = evidence_dir.parent / "work"

        # Use LLM to determine execution context (image and whether to run on host)
        docker_image, run_on_host = determine_execution_context(step.action, self.sandbox.config.sandbox_image)
        
        # If LLM determines it should run on host, execute on host
        if run_on_host:
            stdout, stderr = self._execute_on_host(step.action, work_dir)
            summary = stdout.strip() or stderr.strip() or "Command executed on host."
            return ActionResult(
                ok=True,
                summary=summary,
                evidence={"stdout": stdout, "stderr": stderr, "summary": summary, "execution_type": "host", "docker_image": docker_image},
            )

        # Check if this is a service setup command that should run on host (fallback pattern matching)
        if self._is_service_setup_command(step.action):
            stdout, stderr = self._execute_on_host(step.action, work_dir)
            summary = stdout.strip() or stderr.strip() or "Service setup completed."
            return ActionResult(
                ok=True,
                summary=summary,
                evidence={"stdout": stdout, "stderr": stderr, "summary": summary, "execution_type": "host"},
            )

        # For sandbox execution, restart container with new image if needed
        if self.sandbox._current_image != docker_image:
            try:
                self.sandbox.stop_container()
                self.sandbox.start_container(evidence_dir.parent / "run", work_dir, network=command_needs_network(command), image=docker_image)
            except (subprocess.SubprocessError, OSError, ValueError) as e:
                logger.error(f"Failed to restart container with image {docker_image}: {e}")
                # Fall back to current container if restart fails

        # Execute in sandbox for test scripts
        stdout, stderr = self.sandbox.run(script, evidence_dir, work_dir, network=command_needs_network(command))
        summary = stdout.strip() or stderr.strip() or "Execution completed."
        return ActionResult(
            ok=True,
            summary=summary,
            evidence={"stdout": stdout, "stderr": stderr, "summary": summary, "execution_type": "sandbox", "docker_image": docker_image},
        )
