"""Ask the planner for a Python snippet and run it in the sandbox."""

from pathlib import Path

from aqe.cli_runtime.sandbox import DockerSandbox
from aqe.llm import Planner, command_from_intent, command_needs_network
from aqe.state import ActionResult, TestStep


class CLISubsystem:
    def __init__(self, planner: Planner, sandbox: DockerSandbox) -> None:
        self.planner = planner
        self.sandbox = sandbox

    def execute_runtime_action(self, step: TestStep, evidence_dir: Path) -> ActionResult:
        checks = step.verifications or [step.assertion]
        intent = step.action + "\n" + "\n".join(f"Assertion: {item}" for item in checks)
        script = self.planner.script_for(intent)
        command = command_from_intent(intent) or ""
        stdout, stderr = self.sandbox.run(script, evidence_dir, network=command_needs_network(command))
        summary = stdout.strip() or stderr.strip() or "Execution completed."
        return ActionResult(
            ok=True,
            summary=summary,
            evidence={"stdout": stdout, "stderr": stderr, "summary": summary},
        )
