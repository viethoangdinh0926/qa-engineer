"""Ask the planner for a Python snippet and run it in the host environment."""

import logging
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from aqe.llm import Planner
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


def ensure_tool_available(tool_name: str) -> bool:
    """Check if a tool is available and try to install it if not."""
    # Check if tool is available
    if shutil.which(tool_name):
        return True
    
    # Try to install the tool
    try:
        logger.info(f"Tool {tool_name} not found, attempting to install...")
        
        # Determine installation command based on tool
        install_commands = {
            "curl": ["apt-get", "update", "&&", "apt-get", "install", "-y", "curl"],
            "wget": ["apt-get", "update", "&&", "apt-get", "install", "-y", "wget"],
            "jq": ["apt-get", "update", "&&", "apt-get", "install", "-y", "jq"],
            "docker": ["curl", "-fsSL", "https://get.docker.com", "|", "sh"],
        }
        
        install_cmd = install_commands.get(tool_name)
        if install_cmd:
            # Try with sudo first
            for cmd_prefix in [["sudo"], []]:
                try:
                    full_cmd = cmd_prefix + install_cmd
                    subprocess.run(
                        full_cmd,
                        capture_output=True,
                        text=True,
                        timeout=300,
                        check=False,
                    )
                    # Check if tool is now available
                    if shutil.which(tool_name):
                        logger.info(f"Tool {tool_name} installed successfully")
                        return True
                except (subprocess.TimeoutExpired, OSError):
                    continue
        
        logger.warning(f"Could not install tool {tool_name}")
        return False
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        logger.warning(f"Failed to install tool {tool_name}: {exc}")
        return False


def ensure_python_available() -> bool:
    """Check if Python is available and try to install it if not."""
    if shutil.which("python3") or shutil.which("python"):
        return True
    
    try:
        logger.info("Python not found, attempting to install...")
        # Try to install Python
        for cmd_prefix in [["sudo"], []]:
            try:
                subprocess.run(
                    cmd_prefix + ["apt-get", "update", "&&", "apt-get", "install", "-y", "python3"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                    shell=True,
                )
                if shutil.which("python3") or shutil.which("python"):
                    logger.info("Python installed successfully")
                    return True
            except (subprocess.TimeoutExpired, OSError):
                continue
        return False
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        logger.warning(f"Failed to install Python: {exc}")
        return False


class CLISubsystem:
    def __init__(self, planner: Planner, work_dir: Path) -> None:
        self.planner = planner
        self.work_dir = work_dir

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
        ]
        command_lower = command.lower()
        return any(pattern in command_lower for pattern in service_patterns)

    def _execute_on_host(self, command: str, work_dir: Path) -> tuple[str, str, int]:
        """Execute a command on the host machine. Returns stdout, stderr, and exit code."""
        try:
            # Clean the command string - remove any problematic characters
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
                return "", "No commands to execute", 1

            all_stdout = []
            all_stderr = []
            final_exit_code = 0

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
                final_exit_code = result.returncode
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
                        # Use the last command's exit code, or fail if any command failed
                        if result.returncode != 0:
                            final_exit_code = result.returncode
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
                            if result.returncode != 0:
                                final_exit_code = result.returncode
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
                            if result.returncode != 0:
                                final_exit_code = result.returncode

            stdout = "\n".join(all_stdout)
            stderr = "\n".join(all_stderr)
            return stdout, stderr, final_exit_code
        except subprocess.TimeoutExpired:
            return "", "Command timed out on host machine", 124  # Standard timeout exit code
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            logger.error(f"Host execution error: {e!s}, command was: {command!r}")
            return "", f"Host execution failed: {e!s}", 1

    def execute_runtime_action(self, step: TestStep, evidence_dir: Path) -> ActionResult:
        # Only use the action for script generation, not the verifications
        # Verifications are checked separately after execution
        intent = step.action
        
        # Clean up the action: replace periods used as command separators with &&
        # This handles cases where LLM uses "." instead of "&&" or ";"
        # Pattern: "command1. command2" -> "command1 && command2"
        # But avoid matching periods after shell operators like &, |, ;, &&, ||
        intent = re.sub(r'(?<![&|;])\.\s+(?=\S)', ' && ', intent)
        
        script = self.planner.script_for(intent)
        # work_dir is already set to runs/<run_id>/work by the service
        # Use it directly without creating nested paths
        work_dir = self.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

        # Check if the script is a direct CLI command (not Python code)
        is_direct_cli = (
            not script.strip().startswith(("import", "from", "def", "class", "#", '"', "'")) and
            not any(keyword in script for keyword in ["import ", "from ", "def ", "class ", "print("])
        )

        if is_direct_cli:
            # Execute as direct CLI command
            stdout, stderr, exit_code = self._execute_on_host(script, work_dir)
            summary = stdout.strip() or stderr.strip() or "Command executed."
            return ActionResult(
                ok=(exit_code == 0),
                summary=summary,
                evidence={
                    "stdout": stdout,
                    "stderr": stderr,
                    "summary": summary,
                    "execution_type": "host",
                    "exit_code": exit_code
                },
            )
        else:
            # Execute as Python script
            # Ensure Python is available
            if not ensure_python_available():
                return ActionResult(
                    ok=False,
                    summary="Python is not available and could not be installed",
                    evidence={"stdout": "", "stderr": "Python not available", "execution_type": "host"},
                )
            
            # Write script to file
            script_path = work_dir / "script.py"
            script_path.write_text(script, encoding="utf-8")
            
            # Execute Python script
            try:
                python_cmd = "python3" if shutil.which("python3") else "python"
                result = subprocess.run(
                    [python_cmd, str(script_path)],
                    cwd=str(work_dir),
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                stdout = result.stdout
                stderr = result.stderr
                summary = stdout.strip() or stderr.strip() or "Python script executed."
                return ActionResult(
                    ok=result.returncode == 0,
                    summary=summary,
                    evidence={
                        "stdout": stdout,
                        "stderr": stderr,
                        "summary": summary,
                        "execution_type": "host",
                        "exit_code": result.returncode
                    },
                )
            except subprocess.TimeoutExpired:
                return ActionResult(
                    ok=False,
                    summary="Python script timed out",
                    evidence={"stdout": "", "stderr": "Script timed out", "execution_type": "host", "exit_code": 124},
                )
            except (subprocess.SubprocessError, OSError) as e:
                return ActionResult(
                    ok=False,
                    summary=f"Python script execution failed: {e}",
                    evidence={"stdout": "", "stderr": str(e), "execution_type": "host", "exit_code": 1},
                )
