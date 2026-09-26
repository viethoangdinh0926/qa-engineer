"""Planners that turn a specification into a test matrix or a rejection."""

import json
import logging
import re
import shlex
from pathlib import PurePath
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from aqe.config import EngineConfig
from aqe.state import CodingAction, GUIAction, PlanResult, TestPhase, TestStep

logger = logging.getLogger(__name__)

NONSENSE_SPEC = "Hello, this is not a test specification."

_STEP_LINE = re.compile(
    r"^(?:(\d+)\.\s*)?(GUI|CLI|CODING)(?:\s+(browser|desktop))?\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)
_ASSERTION_LINE = re.compile(r"^assertion\s*:\s*(.+)$", re.IGNORECASE)


def _reject(reason: str) -> PlanResult:
    return PlanResult(
        accepted=False,
        reason=reason,
        reason_code="not_a_test_plan",
    )


def _parse_labeled_steps(specification: str) -> list[TestStep]:
    steps: list[TestStep] = []
    pending: dict[str, object] | None = None

    def flush(assertion: str | None = None) -> None:
        nonlocal pending
        if pending is None:
            return
        if assertion:
            pending["assertion"] = assertion
        if pending.get("assertion"):
            steps.append(TestStep.model_validate(pending))
        pending = None

    for raw_line in specification.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        assertion = _ASSERTION_LINE.match(line)
        if assertion and pending is not None:
            flush(assertion.group(1).strip())
            continue
        match = _STEP_LINE.match(line)
        if not match:
            continue
        flush()
        number = int(match.group(1) or len(steps) + 1)
        interface = match.group(2).upper()
        driver = (match.group(3) or "").lower() or None
        if interface == "GUI" and driver is None:
            driver = "browser"
        if interface in {"CLI", "CODING"}:
            driver = None
        pending = {
            "step": number,
            "interface": interface,
            "gui_driver": driver,
            "action": match.group(4).strip(),
            "assertion": "",
        }
    flush()
    return steps


def _validate_steps(steps: list[TestStep]) -> str | None:
    if not steps:
        return "no verifiable GUI, CLI, or CODING actions were found"
    seen: set[int] = set()
    for step in steps:
        problem = step.validate_shape()
        if problem:
            return problem
        if step.step in seen:
            return f"step {step.step} is duplicated"
        seen.add(step.step)
    return None


class Planner(Protocol):
    def plan(self, specification: str) -> PlanResult:
        """Return a matrix or a rejection. Do not execute anything."""

    def script_for(self, intent: str) -> str:
        """Return a Python snippet for a CLI step."""

    def fix_step(self, step: dict, error: str, retry_count: int) -> dict:
        """Return a fixed version of a failed step."""

    def update_coding_context(self, coding_instructions: list[dict]) -> None:
        """Update the planner's context with coding instructions."""

    def refine_plan(self, current_phases: list[TestPhase], current_steps: list[TestStep], 
                   user_feedback: str, chat_history: list[dict]) -> tuple[list[TestPhase], list[TestStep], str]:
        """Refine the plan based on user feedback. Returns (phases, steps, reasoning)."""


_AGENT_CAPABILITIES = (
    "The agent can use two capabilities: a web browser (open a page, type into a field, click a button, press a key) "
    "and host CLI tools (shell commands). "
    "Desktop input and a coding agent are not available. "
    "Every phase must be executable with the browser or a CLI command. "
    "If the request cannot be tested with those capabilities, set accepted to false and explain that mismatch in reason. "
)

_PLAN_SYSTEM = (
    _AGENT_CAPABILITIES
    + "You are the planner, not the executor. Another system will run the phases you write: "
    "GUI phases in the browser, and CLI phases with host CLI tools. "
    "Do not reject a request because you cannot click, browse, read files, or write code yourself. "
    "Turn the testing request into one linear pipeline of testing phases. "
    "Reply with one JSON object only, with keys accepted, reason, and phases. "
    "Each phase is a chain of operations followed by the verifications of those operations. "
    "A phase has phase (an integer), name, depends_on (a list of earlier phase numbers, or empty), "
    "interface (GUI or CLI), gui_driver (browser or null), operations, and verifications. "
    "GUI operations are objects with action goto, type, click, or press, plus text and selector {role, name} when needed. "
    "CLI operations are strings. "
    "GUI verifications are questions about the page after the operations. "
    "CLI verifications are questions about the command output. "
    "Write every verification as one or two complete sentences. Be specific and verbose. "
    "Keep the original meaning of the user's check. Do not add a condition they did not ask for, and do not drop one they did. "
    "Do not shorten a check into contains:, json:, or status:. "
    "If the user says a command returns nothing or mentions its output, say whether stdout is empty. "
    "Do not add a stdout content check unless the user mentions stdout, standard output, or the command output. "
    "Mention stderr only when the user mentions an error, issue, exception, warning, failure, traceback, or stderr. "
    "Example: 'Run ls. Verify that it returns nothing.' has verification "
    "'The ls command returns nothing. stdout is empty. A successful exit code does not satisfy this check.' "
    "Example: 'Run curl and verify that it reports an error.' has verification "
    "'The curl command reports an error.' It does not mention stdout. "
    "IMPORTANT: Never use environment variables ($VAR) in CLI operations. Use concrete values instead. "
    "For example, instead of 'curl http://localhost:$PORT', use 'curl http://localhost:8080'. "
    "Instead of 'docker port $container_id 5000', write a step that gets the actual container ID and uses it directly. "
    "IMPORTANT: Never use Docker or any containerizing techniques (docker run, docker-compose, kubectl, podman, etc.) for any test step. "
    "All CLI operations must run directly in the host environment without containers. "
    "Do not instruct the agent to start containers, use docker commands, or containerize any part of the test execution. "
    "IMPORTANT: When starting a long-running service (e.g., using nohup, backgrounding with &, or starting a server), "
    "you MUST include a verification step to confirm the service started successfully and is still running. "
    "For example, after 'nohup python app.py > service.log 2>&1 &', add a verification like "
    "'A python process running app.py is present, indicating that the API service has started successfully.' "
    "Or check the log file: 'The service.log file exists and contains no error messages, indicating the service started without crashing.' "
    "Or check if the service is listening: 'The service is listening on the expected port, indicating it started successfully.' "
    "This ensures that service startup failures or crashes are detected. "
    "Every phase needs at least one verification. "
    "Put a phase that needs another phase's result after that phase, and list it in depends_on. "
    "One page visit is one GUI phase. Navigation, typing, and the click are that phase's operations. "
    "The page check is a verification on that same phase, not a later phase. "
    "A request to open a page, type into a named field, click a named button, and then check the page text "
    "is accepted true as one GUI phase. It is not a contradiction. "
    "Use the field name and button name from the request. "
    'Example: open http://localhost:8765, type ada into Username, click Commit, and verify the page contains registered becomes '
    '{"accepted": true, "reason": null, "phases": [{"phase": 1, "name": "Register ada", "depends_on": [], '
    '"interface": "GUI", "gui_driver": "browser", "operations": ['
    '{"action": "goto", "text": "http://localhost:8765"}, '
    '{"action": "type", "selector": {"role": "textbox", "name": "Username"}, "text": "ada"}, '
    '{"action": "click", "selector": {"role": "button", "name": "Commit"}}], '
    '"verifications": ["The page contains registered"]}]} '
    "Set accepted false only when the request itself is not a test, contradicts itself, "
    "or asks for a check whose result nothing in the request produces. "
    "When accepted is false, leave phases empty and explain that request problem in reason. "
    "Do not reject because a page check comes after the click that produces the page."
)

_REPAIR_SYSTEM = (
    _AGENT_CAPABILITIES
    + "Revise the testing plan so every phase is executable with the browser or a CLI command. "
    "Reply with one JSON object only, with keys accepted, reason, and phases. "
    "Use the same phase schema: phase, name, depends_on, interface, gui_driver, operations, coding_operations, and verifications. "
    "verifications must be strings. "
    "Every check the request asks for must appear as a verification written as one or two complete sentences. "
    "Be verbose and keep the original meaning. Do not shorten a check into contains:, json:, or status:. "
    "A command that should return nothing is described as empty stdout, not as an exit code. "
    "Do not add a stdout content check unless the user mentions stdout, standard output, or the command output. "
    "Mention stderr only when the user mentions an error, issue, exception, warning, failure, traceback, or stderr. "
    "A page check belongs on the GUI phase whose operations produce that page. Never make the check its own phase. "
    "A file, webhook, or command check is a later CLI phase whose depends_on lists the phase that produced it. "
    "Coding operations (create_file, update_file, review_code, execute_code) belong in CODING phases. "
    "CODING operations ARE valid test operations. They are not 'development activities' to be rejected. "
    "IMPORTANT: When a phase has interface CODING, you MUST include coding_operations with the actual file operations. "
    "Do not create a CODING phase with only verifications and no coding_operations. "
    "Keep phases in an order that respects those dependencies. "
    "You are the planner, not the executor. Do not reject a request because you cannot browse, read files or write code. "
    "Opening a page, typing into a named field, clicking a named button, and checking the resulting page "
    "is one accepted GUI phase. Use those names in the selectors. "
    "Phases without verifications are allowed for setup/preparation steps. "
    "If all phases have no verifications, the test passes if all operations succeed. "
    "Set accepted false only when the request itself is not a test or contradicts itself."
)

_REJECT_SYSTEM = (
    "Explain why this testing request itself cannot be planned. Plain sentences, not JSON. "
    "Describe only a problem in the request: it is not a test, two requirements contradict each other, "
    "or a check needs a result the request never produces. "
    "CODING operations (create_file, update_file, review_code, execute_code) ARE valid test operations. "
    "Do not reject a request just because it involves writing code or creating files. "
    "Do not call it a contradiction when the request opens a page, fills a field, clicks a button, "
    "and then checks the page text. That is one GUI phase, with the check after the click. "
    "Do not talk about a verification phase that has no operations."
)

_OPS_SYSTEM = (
    "Reply with one JSON object only, no markdown. "
    '{"operations": [{"step": 1, "operations": ['
    '{"action": "goto", "text": "http://127.0.0.1:8765"}, '
    '{"action": "type", "selector": {"role": "textbox", "name": "Username"}, "text": "ada"}, '
    '{"action": "click", "selector": {"role": "button", "name": "Register"}}'
    "]}]} "
    "Include every control the phase names, including the submit click. "
    "Use the field name and button name from the phase text. "
    "If the phase says Username, the textbox name is Username. "
    "If the phase says Commit, the button name is Commit. "
    "Use the given target URL for goto."
)

_SCRIPT_SYSTEM = (
    "Reply with Python only, no markdown and no functions. "
    "Perform the operation in the user message and print its result to stdout. "
    "Do not read /evidence unless the operation names a file there. "
    "IMPORTANT: Always prefer direct CLI commands over Python scripts when possible. "
    "Use common CLI tools like curl, wget, jq, grep, sed, awk, etc. "
    "Only use Python scripts when the operation requires complex logic, loops, or data processing that CLI tools cannot handle. "
    "IMPORTANT: Never use environment variables ($VAR) in commands. Use concrete values instead. "
    "For example, instead of 'curl http://localhost:$PORT', use 'curl http://localhost:8080'. "
    "Instead of 'docker port $container_id 5000', use the actual container ID. "
    "IMPORTANT: Never use Docker or any containerizing techniques (docker run, docker-compose, kubectl, podman, etc.) for any operation. "
    "All commands must run directly in the host environment without containers."
)

_FIX_STEP_SYSTEM = (
    "You are fixing a failed test step. "
    "Review the failed step and the error message, then determine if this is a hard failure that cannot be fixed by retrying. "
    "Hard failures include: "
    "- Port conflicts (e.g., 'Address already in use', 'port is in use') "
    "- Missing dependencies that cannot be installed (e.g., 'No module named X' when X is not available) "
    "- Permission errors that cannot be resolved (e.g., 'Permission denied' for system files) "
    "- Network errors indicating the service is not running on the expected port "
    "- File not found errors for files that don't exist and cannot be created "
    "- Service startup failures that indicate the service cannot run (e.g., 'ModuleNotFoundError' for required modules) "
    "If the error is a hard failure, set 'should_retry' to false and explain why in the judgment. "
    "If the error can be fixed (e.g., wrong file path, wrong command name, syntax error, timeout too short), "
    "set 'should_retry' to true and provide the corrected step. "
    "Reply with one JSON object only. "
    "If should_retry is true, include the corrected step with these fields: step, interface, gui_driver, action, assertion, verifications, operations, coding_operations. "
    "Keep the same step number and interface. "
    "Fix the action, operations, or coding_operations to address the error. "
    "Common fixes: "
    "- If file not found: correct the file path or create the file first "
    "- If command not found: use the correct command name or install the tool "
    "- If permission denied: add sudo or use a different approach "
    "- If syntax error: fix the command syntax "
    "- If timeout: add a longer timeout or break into smaller steps "
    "- If dependency missing: add a step to install the dependency "
    "IMPORTANT: Never use Docker or any containerizing techniques (docker run, docker-compose, kubectl, podman, etc.) for the fix. "
    "All operations must run directly in the host environment without containers. "
    "Return JSON with keys: 'should_retry' (true or false), 'step' (the corrected step, only if should_retry is true), 'judgment' (explanation)."
)

_REFINE_PLAN_SYSTEM = (
    _AGENT_CAPABILITIES
    + "You are a test planning assistant. Users may ask you questions about the current plan or request changes to it.\n\n"
    "If the user asks a question (e.g., 'Why did you include this step?', 'Explain phase 2', 'What does this verification check?'), "
    "provide a clear, helpful answer about the current plan. Return JSON with:\n"
    "{\n"
    "  \"type\": \"answer\",\n"
    "  \"answer\": \"your explanation here\"\n"
    "}\n\n"
    "If the user requests changes (e.g., 'Add a verification for X', 'Remove step 3', 'Change the approach'), "
    "refine the plan accordingly. Return JSON with:\n"
    "{\n"
    "  \"type\": \"plan_update\",\n"
    "  \"phases\": [\n"
    "    {\n"
    "      \"phase\": integer (phase number),\n"
    "      \"name\": string (phase name),\n"
    "      \"interface\": \"GUI\" or \"CLI\",\n"
    "      \"gui_driver\": \"browser\", \"desktop\", or null,\n"
    "      \"depends_on\": [list of phase numbers],\n"
    "      \"operations\": [\n"
    "        {\n"
    "          \"action\": \"click\", \"type\", \"press\", or \"goto\",\n"
    "          \"coordinate\": [x, y] or null,\n"
    "          \"selector\": {\"role\": \"textbox\", \"name\": \"field_name\"} or null,\n"
    "          \"text\": string or null\n"
    "        }\n"
    "      ],\n"
    "      \"coding_operations\": [\n"
    "        {\n"
    "          \"action\": \"create_file\", \"update_file\", \"review_code\", or \"execute_code\",\n"
    "          \"file_path\": string (file path),\n"
    "          \"content\": string (file content),\n"
    "          \"description\": string (description)\n"
    "        }\n"
    "      ],\n"
    "      \"verifications\": [list of verification strings]\n"
    "    }\n"
    "  ],\n"
    "  \"steps\": [\n"
    "    {\n"
    "      \"step\": integer (step number),\n"
    "      \"interface\": \"GUI\" or \"CLI\",\n"
    "      \"gui_driver\": \"browser\", \"desktop\", or null,\n"
    "      \"action\": string (action description),\n"
    "      \"assertion\": string (assertion),\n"
    "      \"verifications\": [list of verification strings],\n"
    "      \"operations\": [\n"
    "        {\n"
    "          \"action\": \"click\", \"type\", \"press\", or \"goto\",\n"
    "          \"coordinate\": [x, y] or null,\n"
    "          \"selector\": {\"role\": \"textbox\", \"name\": \"field_name\"} or null,\n"
    "          \"text\": string or null\n"
    "        }\n"
    "      ],\n"
    "      \"coding_operations\": [\n"
    "        {\n"
    "          \"action\": \"create_file\", \"update_file\", \"review_code\", or \"execute_code\",\n"
    "          \"file_path\": string (file path),\n"
    "          \"content\": string (file content),\n"
    "          \"description\": string (description)\n"
    "        }\n"
    "      ]\n"
    "    }\n"
    "  ],\n"
    "  \"reasoning\": string (explanation of changes)\n"
    "}\n\n"
    "CRITICAL: For plan updates, use 'phase' (not 'id') for the phase number. Always include 'interface' field. "
    "GUI operations are objects. CLI operations are command strings. "
    "Do not add a coding phase or a desktop phase."
)

_NETWORK_COMMANDS = frozenset({"curl", "wget", "ping", "dig", "nslookup", "host", "nc", "ncat", "pip", "pip3", "npm", "npm install", "apt", "apt-get", "yum", "dnf"})


def _operations_from_action(action: dict) -> list[dict]:
    nested = action.get("operations")
    if isinstance(nested, list):
        return nested
    operations: list[dict] = []
    goto = action.get("goto")
    if isinstance(goto, str) and goto:
        operations.append({"action": "goto", "text": goto})
    selector = action.get("selector") if isinstance(action.get("selector"), dict) else None
    text = action.get("text") if isinstance(action.get("text"), str) else None
    kind = action.get("action") or action.get("type")
    if selector and text and kind in {None, "text", "type"}:
        operations.append({"action": "type", "selector": selector, "text": text})
    elif selector and kind == "click":
        operations.append({"action": "click", "selector": selector, "text": text})
    return operations


def _coerce_steps(steps: object) -> list[dict]:
    if not isinstance(steps, list):
        logger.warning(f"Steps is not a list: {type(steps)}")
        return []
    coerced: list[dict] = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            logger.warning(f"Step {index} is not a dict: {type(step)}")
            continue
        item = dict(step)
        number = item.get("step")
        action = item.get("action")
        described_action = action if isinstance(action, dict) else None
        if isinstance(number, str) and number.isdigit():
            item["step"] = int(number)
        elif not isinstance(number, int):
            item["step"] = index
            if isinstance(number, str) and described_action is not None:
                item["action"] = number
        
        # Handle operations field
        operations = item.get("operations")
        interface = item.get("interface", "").upper()
        action = item.get("action")
        
        # For CLI steps, operations should not contain strings - those should be in action
        if interface == "CLI" and isinstance(operations, list):
            # If operations contains strings, this is likely a mistake by the LLM
            # Move the operations to the action field
            if operations and all(isinstance(op, str) for op in operations):
                # Combine the operations into a single action
                item["action"] = " && ".join(operations)
                item["operations"] = None  # Clear operations for CLI steps
                logger.info(f"Moved CLI operations to action for step {index}")
            elif operations and any(isinstance(op, str) for op in operations):
                # Mixed types - keep only GUIAction objects, warn about strings
                gui_ops = [op for op in operations if isinstance(op, dict)]
                if gui_ops:
                    item["operations"] = gui_ops
                else:
                    item["operations"] = None
                logger.warning(f"Mixed operations in CLI step {index}, kept only GUI actions")
            elif operations and all(isinstance(op, dict) for op in operations):
                # All operations are dicts - might be CLI commands in dict format
                # Check if they have 'command' field instead of 'action'
                for op in operations:
                    if isinstance(op, dict) and "command" in op and "action" not in op:
                        # This is likely a CLI command in dict format
                        # Convert to string and add to action
                        if not item.get("action"):
                            item["action"] = op.get("command", "")
                        else:
                            item["action"] += f" && {op.get('command', '')}"
                item["operations"] = None
                logger.info(f"Converted CLI dict operations to action for step {index}")
        
        # For GUI steps, operations should be GUIAction objects
        if interface == "GUI" and (isinstance(action, dict) or described_action is not None):
            source = described_action if described_action is not None else action
            item["operations"] = item.get("operations") or _operations_from_action(source)
            if not isinstance(item.get("action"), str):
                item["action"] = "Perform the described action."
        
        # For CODING steps, operations should not be used
        if interface == "CODING":
            item["operations"] = None
        
        coerced.append(item)
    logger.info(f"Coerced {len(coerced)} steps from {len(steps) if isinstance(steps, list) else 0} input steps")
    return coerced


def _best_assertion(steps: list[TestStep]) -> str:
    page_checks: list[str] = []
    for step in steps:
        assertion = step.assertion.strip()
        if not assertion.startswith("contains:"):
            continue
        value = assertion.split(":", 1)[1].strip().strip("\"'")
        if value.startswith("http"):
            continue
        page_checks.append(f"contains:{value}")
    if page_checks:
        return page_checks[-1]
    return steps[-1].assertion


def _collapse_gui_sequences(steps: list[TestStep]) -> list[TestStep]:
    """One browser visit is one step. A later check of the same page is its assertion."""
    collapsed: list[TestStep] = []
    buffer: list[TestStep] = []

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        if len(buffer) == 1:
            collapsed.append(buffer[0])
        else:
            operations = [operation for step in buffer for operation in step.operations]
            action = ". ".join(step.action.strip().rstrip(".") for step in buffer if step.action.strip())
            collapsed.append(
                buffer[0].model_copy(
                    update={
                        "action": action,
                        "assertion": _best_assertion(buffer),
                        "operations": operations,
                    }
                )
            )
        buffer = []

    for step in steps:
        if step.interface == "GUI":
            buffer.append(step)
            continue
        flush()
        collapsed.append(step)
    flush()
    return [step.model_copy(update={"step": index}) for index, step in enumerate(collapsed, start=1)]


def _missing_operations(steps: list[TestStep]) -> str | None:
    for step in steps:
        if step.interface == "GUI" and not step.operations:
            return f"step {step.step} has no GUI operations"
        if step.interface == "CODING" and not step.coding_operations:
            return f"step {step.step} has no coding operations"
        if step.interface == "CLI" and not step.operation_notes and not step.action.strip():
            return f"step {step.step} has no CLI operations"
    return None


def _operation_label(operation: dict | GUIAction) -> str:
    if isinstance(operation, GUIAction):
        action = operation.action
        text = operation.text or ""
        selector = operation.selector or {}
    else:
        action = str(operation.get("action") or "")
        text = str(operation.get("text") or "")
        selector = operation.get("selector") if isinstance(operation.get("selector"), dict) else {}
    name = str(selector.get("name") or "")
    if action == "goto":
        return f"goto {text}".strip()
    if action == "type":
        target = f" into {name}" if name else ""
        return f"type {text}{target}".strip()
    if action == "click":
        return f"click {name}".strip()
    if action == "press":
        return f"press {text or 'Enter'}".strip()
    return action or text


def _split_operations(raw: object) -> tuple[list[str], list[dict]]:
    notes: list[str] = []
    actions: list[dict] = []
    items = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
    for item in items:
        if isinstance(item, str) and item.strip():
            notes.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        kind = item.get("action")
        if kind in {"click", "type", "press", "goto"}:
            actions.append(item)
            notes.append(_operation_label(item))
            continue
        text = item.get("description") or item.get("name") or item.get("operation")
        if isinstance(text, str) and text.strip():
            notes.append(text.strip())
    return notes, actions


def _string_list(raw: object) -> list[str]:
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    values: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            values.append(item.strip())
            continue
        if isinstance(item, dict):
            text = _verification_text(item)
            if text:
                values.append(text)
    return values


def _verification_text(item: dict) -> str | None:
    for key in ("question", "description", "assertion", "check", "verification"):
        text = item.get(key)
        if isinstance(text, str) and text.strip():
            return text.strip()
    text = item.get("text") or item.get("expected")
    if not isinstance(text, str) or not text.strip():
        return None
    kind = str(item.get("type") or "").lower()
    if kind in {"page_check", "contains"} and not text.strip().startswith("contains:"):
        return f"The page contains {text.strip()}"
    return text.strip()


_INCOMPLETE_REASON = ("assum", "implicit", "omit", "skip", "not included", "no interface", "left out", "dropped")
_REFUSED_TO_PLAN = (
    "do not have the capability",
    "cannot execute",
    "cannot construct",
    "as an llm",
    "i cannot",
    "i do not",
    "cannot browse",
    "cannot access",
)


def _looks_incomplete(reason: str | None) -> bool:
    if not reason:
        return False
    text = reason.lower()
    return any(word in text for word in _INCOMPLETE_REASON)


def _refused_to_plan(reason: str | None) -> bool:
    if not reason:
        return False
    text = reason.lower()
    return any(phrase in text for phrase in _REFUSED_TO_PLAN)


def _structural_excuse(reason: str | None) -> bool:
    if not reason:
        return False
    text = reason.lower()
    return any(
        phrase in text
        for phrase in (
            "no operations",
            "no verifications",
            "standalone",
            "without any preceding",
            "phase structure",
            "verification phase",
            "separation between",
            "verification step",
        )
    )


def _phase_number(raw: object, fallback: int) -> int:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.isdigit():
        return int(raw)
    return fallback


def _depends_on(raw: object) -> list[int]:
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, str) and raw.isdigit():
        return [int(raw)]
    if not isinstance(raw, list):
        return []
    numbers: list[int] = []
    for item in raw:
        if isinstance(item, int):
            numbers.append(item)
        elif isinstance(item, str) and item.isdigit():
            numbers.append(int(item))
    return list(dict.fromkeys(numbers))


def _coerce_phases(raw: list) -> tuple[list[TestPhase], str | None]:
    phases: list[TestPhase] = []
    problems: list[str] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            problems.append(f"Phase {index} is not an object, so its operations and verifications cannot be read.")
            continue
        number = _phase_number(item.get("phase") or item.get("id"), index)
        name = str(item.get("name") or item.get("title") or "").strip()
        notes, actions = _split_operations(item.get("operations") or item.get("operation"))
        verifications = _string_list(item.get("verifications") or item.get("verification") or item.get("assertions"))
        if not name:
            name = notes[0] if notes else f"Phase {number}"
        interface = str(item.get("interface") or "").strip().upper()
        driver = item.get("gui_driver")
        driver_name = str(driver).strip().lower() if isinstance(driver, str) and driver.strip() else None

        # Handle coding operations
        coding_operations_raw = item.get("coding_operations") or []
        coding_operations = []
        if isinstance(coding_operations_raw, list):
            for op in coding_operations_raw:
                if isinstance(op, dict):
                    # Fix common LLM mistakes: rename 'operation' to 'action'
                    if "operation" in op and "action" not in op:
                        op = dict(op)
                        op["action"] = op.pop("operation")
                    # Fix common LLM mistakes: rename 'type' to 'action'
                    if "type" in op and "action" not in op:
                        op = dict(op)
                        op_type = op.pop("type")
                        # Map common type values to action values
                        type_to_action = {
                            "write_file": "create_file",
                            "create_file": "create_file",
                            "update_file": "update_file",
                            "edit_file": "update_file",
                            "modify_file": "update_file",
                            "review_code": "review_code",
                            "execute_code": "execute_code",
                            "execute_command": "execute_code",
                            "run_code": "execute_code",
                            "make_executable": "execute_code",  # Treat as execute_code with chmod
                        }
                        op["action"] = type_to_action.get(op_type, op_type)
                    # Fix common LLM mistakes: rename 'execute_command' to 'execute_code'
                    if op.get("action") == "execute_command":
                        op = dict(op)
                        op["action"] = "execute_code"
                    # Fix common LLM mistakes: rename 'command' to 'action' (for execute_code)
                    if "command" in op and "action" not in op:
                        op = dict(op)
                        op["action"] = "execute_code"
                        # Keep the command in a separate field if needed
                        if "command" in op:
                            op["content"] = op.pop("command")
                    try:
                        coding_operations.append(CodingAction.model_validate(op))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(f"Failed to validate coding operation: {exc}")

        # Auto-detect interface if not specified
        if interface not in {"GUI", "CLI", "CODING"}:
            if coding_operations:
                interface = "CODING"
            elif driver_name in {"browser", "desktop"} or actions:
                interface = "GUI"
            elif any(check.startswith(("json:", "status:")) for check in verifications):
                interface = "CLI"
            else:
                problems.append(
                    f"Phase {number} ({name}) does not say whether its operations run through the GUI, CLI, or CODING. "
                    "The pipeline cannot place that phase because the driver is unknown."
                )
                continue

        if interface == "GUI" and driver_name not in {"browser", "desktop"}:
            driver_name = "browser"
        if interface in {"CLI", "CODING"}:
            driver_name = None
        if interface == "CLI" and not verifications and notes:
            verifications = list(notes)
        if interface == "CLI" and verifications and not notes and not actions:
            notes = [f"Carry out: {item}" for item in verifications]
        if interface == "CODING" and not verifications and notes:
            verifications = list(notes)
        if interface == "CODING" and verifications and not notes and not coding_operations:
            notes = [f"Carry out: {item}" for item in verifications]

        phases.append(
            TestPhase(
                phase=number,
                name=name,
                interface=interface,  # type: ignore[arg-type]
                gui_driver=driver_name,  # type: ignore[arg-type]
                depends_on=_depends_on(item.get("depends_on") or item.get("after")),
                operations=[GUIAction.model_validate(action) for action in actions],
                operation_notes=notes,
                verifications=verifications,
                coding_operations=coding_operations,
            )
        )
    if problems:
        return [], "\n".join(problems)
    return phases, None


_EMPTY_OUTPUT_PHRASES = (
    "returns nothing",
    "return nothing",
    "prints nothing",
    "print nothing",
    "returns empty",
    "no output",
    "outputs nothing",
    "empty output",
    "nothing is returned",
)


def _wants_empty_output(specification: str) -> bool:
    text = specification.lower()
    return any(phrase in text for phrase in _EMPTY_OUTPUT_PHRASES)


def verbalize_verification(specification: str, check: str) -> str:
    """Turn a shortened check into a full statement of the same meaning."""
    text = check.strip()
    lowered = text.lower()
    if lowered.startswith("status:"):
        if _wants_empty_output(specification):
            return (
                "The command returns nothing. stdout is empty. "
                "A successful exit code does not satisfy this check."
            )
        code = text.split(":", 1)[1].strip()
        code = code.removeprefix("CODE=").removeprefix("code=")
        return (
            f"The command exits with status {code}. "
            "This check is about the exit code, not about the text the command prints."
        )
    if lowered.startswith("contains:"):
        value = text.split(":", 1)[1].strip().strip("\"'")
        return f"The result contains the text \"{value}\". No other substitution satisfies this check."
    if lowered.startswith("json:"):
        expression = text.split(":", 1)[1]
        field, separator, expected = expression.partition("=")
        if separator:
            return (
                f"The JSON output has a field \"{field}\" whose value is \"{expected}\". "
                f"The check is json:{field}={expected}."
            )
        return f"The JSON output includes {expression}."
    return text


_STDOUT_CLAUSES = (
    re.compile(r"(?i)\s*(?:and\s+)?returns its output"),
    re.compile(r"(?i)\s*stdout is not empty"),
    re.compile(r"(?i)\s*stdout is empty"),
    re.compile(r"(?i)\s*standard output is not empty"),
    re.compile(r"(?i)\s*standard output is empty"),
)


def omit_unrequested_stdout(specification: str, phases: list[TestPhase]) -> list[TestPhase]:
    """Drop a stdout content check when the request never mentions output."""
    from aqe.judge import mentions_stdout

    if mentions_stdout(specification):
        return phases
    omitted: list[TestPhase] = []
    for phase in phases:
        if phase.interface != "CLI":
            omitted.append(phase)
            continue
        checks = [_without_stdout_claim(item) for item in phase.verifications]
        checks = [item for item in checks if item]
        if not checks:
            checks = ["The command finishes."]
        if checks == list(phase.verifications):
            omitted.append(phase)
            continue
        omitted.append(phase.model_copy(update={"verifications": checks}))
    return omitted


def _without_stdout_claim(check: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", check.strip())
    kept: list[str] = []
    for sentence in sentences:
        cleaned = sentence
        for pattern in _STDOUT_CLAUSES:
            cleaned = pattern.sub("", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ;")
        cleaned = re.sub(r"\s+([.!?])", r"\1", cleaned)
        if cleaned and not re.fullmatch(r"[.!?]+", cleaned):
            kept.append(cleaned)
    return " ".join(kept).strip()


def verbalize_verifications(specification: str, phases: list[TestPhase]) -> list[TestPhase]:
    """Keep each phase check verbose and faithful to the request."""
    verbalized: list[TestPhase] = []
    for phase in phases:
        checks = [verbalize_verification(specification, item) for item in phase.verifications]
        if checks == list(phase.verifications):
            verbalized.append(phase)
            continue
        verbalized.append(phase.model_copy(update={"verifications": checks}))
    return verbalized


def _has_work(phase: TestPhase) -> bool:
    return bool(phase.operations or phase.operation_notes or phase.coding_operations)


def fold_check_phases(phases: list[TestPhase]) -> list[TestPhase]:
    """Attach a check that has no operations to the phase that performed the actions."""
    if len(phases) < 2:
        return phases
    by_id = {phase.phase: phase for phase in phases}
    redirect: dict[int, int] = {}
    for index, phase in enumerate(phases):
        if _has_work(phase) or not phase.verifications or phase.phase in redirect:
            continue
        host_id = None
        for dep in phase.depends_on:
            candidate = by_id.get(redirect.get(dep, dep))
            if candidate and _has_work(candidate):
                host_id = candidate.phase
        if host_id is None:
            for earlier in reversed(phases[:index]):
                earlier_id = redirect.get(earlier.phase, earlier.phase)
                candidate = by_id.get(earlier_id)
                if candidate and _has_work(candidate):
                    host_id = candidate.phase
                    break
        if host_id is None:
            for later in phases[index + 1 :]:
                if _has_work(later):
                    host_id = later.phase
                    break
        if host_id is None:
            continue
        host = by_id[host_id]
        checks = list(dict.fromkeys([*host.verifications, *phase.verifications]))
        by_id[host_id] = host.model_copy(update={"verifications": checks})
        redirect[phase.phase] = host_id
    if not redirect:
        return phases
    folded: list[TestPhase] = []
    for phase in phases:
        if phase.phase in redirect:
            continue
        current = by_id[phase.phase]
        depends: list[int] = []
        for dep in current.depends_on:
            dep = redirect.get(dep, dep)
            if dep != current.phase and dep not in depends:
                depends.append(dep)
        folded.append(current.model_copy(update={"depends_on": depends}))
    return folded


def _validate_phases(phases: list[TestPhase]) -> str | None:
    if not phases:
        return (
            "The plan has no testing phases. A request needs a pipeline of phases, "
            "and each phase needs operations to carry out."
        )
    problems: list[str] = []
    for phase in phases:
        label = f"Phase {phase.phase} ({phase.name})"
        has_operations = bool(phase.operation_notes or phase.operations or phase.coding_operations)
        if not has_operations:
            problems.append(
                f"{label} has no operations. "
                f"A phase has to carry out a chain of operations to be actionable."
            )
        if phase.interface == "CODING":
            problems.append(
                f"{label} is not executable. A coding agent is not available. "
                "Use the browser or a CLI command."
            )
        if phase.interface == "GUI" and phase.gui_driver == "desktop":
            problems.append(
                f"{label} is not executable. Desktop input is not available. "
                "Use the browser or a CLI command."
            )
        if not phase.verifications:
            problems.append(
                f"{label} has no verifications. "
                "Each phase needs a check that says what those operations must produce."
            )
    if not problems:
        return None
    return "\n".join(problems)


def order_phases(phases: list[TestPhase]) -> tuple[list[TestPhase], str | None]:
    """Place phases in dependency order. A cycle or a missing dependency is a rejection."""
    by_id: dict[int, TestPhase] = {}
    for phase in phases:
        if phase.phase in by_id:
            other = by_id[phase.phase].name
            return [], (
                f"Phase number {phase.phase} is used by both {other!r} and {phase.name!r}. "
                "Each phase needs its own number so a later phase can name the phase it depends on."
            )
        by_id[phase.phase] = phase
    dependents: dict[int, list[int]] = {number: [] for number in by_id}
    indegree = {number: 0 for number in by_id}
    for phase in phases:
        for dep in phase.depends_on:
            if dep == phase.phase:
                return [], (
                    f"Phase {phase.phase} ({phase.name}) depends on itself. "
                    "A phase can use the result of an earlier phase, but it cannot require its own result "
                    "before it runs, so this pipeline has no valid starting point."
                )
            if dep not in by_id:
                return [], (
                    f"Phase {phase.phase} ({phase.name}) depends on phase {dep}, but phase {dep} is not in the plan. "
                    "The pipeline cannot be ordered until every dependency is a phase that actually runs "
                    "and produces the result this phase needs."
                )
            dependents[dep].append(phase.phase)
            indegree[phase.phase] += 1
    ready = sorted(number for number, degree in indegree.items() if degree == 0)
    ordered: list[TestPhase] = []
    while ready:
        number = ready.pop(0)
        ordered.append(by_id[number])
        for child in dependents[number]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort()
    if len(ordered) != len(phases):
        stuck = [by_id[number] for number, degree in indegree.items() if degree > 0]
        lines = [
            "The testing phases depend on each other in a circle, so they cannot be placed in a linear order.",
        ]
        for phase in stuck:
            deps = ", ".join(str(dep) for dep in phase.depends_on) or "nothing"
            lines.append(f"Phase {phase.phase} ({phase.name}) depends on {deps}.")
        lines.append(
            "A later phase may use an earlier phase's result. "
            "A phase cannot depend on itself or on a phase that depends back on it. "
            "Name which phase produces the result, and make the consumer depend only on that producer."
        )
        return [], "\n".join(lines)
    id_map = {phase.phase: index for index, phase in enumerate(ordered, start=1)}
    renumbered = [
        phase.model_copy(
            update={
                "phase": index,
                "depends_on": [id_map[dep] for dep in phase.depends_on],
            }
        )
        for index, phase in enumerate(ordered, start=1)
    ]
    return renumbered, None


def _phase_to_step(phase: TestPhase) -> TestStep:
    notes = list(phase.operation_notes) or [_operation_label(operation) for operation in phase.operations]
    checks = list(phase.verifications)
    if phase.interface == "CLI" or phase.interface == "CODING":
        action = ". ".join(notes) if notes else phase.name
    else:
        action = phase.name or ". ".join(notes)
    assertion = checks[0] if len(checks) == 1 else "\n".join(checks)
    
    # Handle operations field - convert CLI commands to action if needed
    operations = list(phase.operations)
    if phase.interface == "CLI" and operations:
        # Check if operations contain CLI commands (dicts with 'command' field)
        cli_commands = []
        gui_actions = []
        for op in operations:
            if isinstance(op, dict):
                if "command" in op:
                    # This is a CLI command, convert to string
                    cli_commands.append(op.get("command", ""))
                elif "action" in op:
                    # This is a GUI action
                    gui_actions.append(op)
            elif isinstance(op, str):
                # String operation - treat as CLI command
                cli_commands.append(op)
        
        if cli_commands:
            # Combine CLI commands into action
            if cli_commands:
                action = " && ".join(cli_commands)
            # For CLI steps, operations should be empty or only GUI actions
            operations = gui_actions if gui_actions else []
    
    # Only set operations for GUI steps; CLI and CODING steps should have empty list
    if phase.interface == "GUI":
        operations = list(phase.operations)
    else:
        operations = []  # Empty list instead of None for CLI/CODING steps
    
    return TestStep(
        step=phase.phase,
        interface=phase.interface,
        gui_driver=phase.gui_driver,
        action=action,
        assertion=assertion,
        operations=operations,
        phase=phase.phase,
        phase_name=phase.name,
        depends_on=list(phase.depends_on),
        operation_notes=notes,
        verifications=checks,
        coding_operations=list(phase.coding_operations),
    )


def _repair_json(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r"\bNone\b", "null", text)
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(
        r"'([^'\\]*(?:\\.[^'\\]*)*)'",
        lambda match: json.dumps(match.group(1)),
        text,
    )
    text = re.sub(r"([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', text)
    return text


def _json_object(content: str) -> dict:
    text = _strip_fence(content)
    start = text.find("{")
    if start < 0:
        raise ValueError("planner output did not match the test plan schema")
    blob = text[start:]
    try:
        value, _ = json.JSONDecoder().raw_decode(blob)
    except json.JSONDecodeError:
        value, _ = json.JSONDecoder().raw_decode(_repair_json(blob))
    if not isinstance(value, dict):
        raise TypeError("planner output did not match the test plan schema")
    return value


def _parse_operations(content: str) -> dict[int, list]:
    payload = _json_object(content)
    rows = payload.get("operations") or payload.get("steps") or []
    parsed: dict[int, list] = {}
    if (
        rows
        and isinstance(rows, list)
        and isinstance(rows[0], dict)
        and rows[0].get("action") in {"goto", "type", "click", "press"}
    ):
        return {1: rows}
    for item in rows:
        if not isinstance(item, dict) or "step" not in item:
            continue
        operations = item.get("operations") or []
        parsed[int(item["step"])] = operations
    return parsed


def _strip_fence(content: str) -> str:
    text = content.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_plan(content: str) -> PlanResult:
    payload = _json_object(content)
    logger.info(f"Plan payload keys: {list(payload.keys())}")
    reason = payload.get("reason")
    reason_text = reason.strip() if isinstance(reason, str) and reason.strip() else None
    raw_phases = payload.get("phases")
    if isinstance(raw_phases, list):
        logger.info(f"Processing {len(raw_phases)} phases")
        phases, problem = _coerce_phases(raw_phases)
        if problem:
            return PlanResult(accepted=False, reason=problem, reason_code="not_a_test_plan")
        accepted = payload.get("accepted")
        if accepted is None:
            accepted = bool(phases)
        return PlanResult(accepted=bool(accepted), reason=reason_text, phases=phases)

    logger.info("Processing steps instead of phases")
    # Try to coerce steps with better error handling
    try:
        payload["steps"] = _coerce_steps(payload.get("steps"))
    except Exception as exc:  # noqa: BLE001
        import traceback
        error_detail = str(exc)
        if hasattr(exc, "errors"):
            error_detail = str(exc.errors())
        logger.error(f"Step coercion failed: {error_detail}\n{traceback.format_exc()}")
        logger.error(f"Raw steps: {payload.get('steps')}")
        return _reject(f"planner output did not match the test plan schema: {error_detail}")
    
    if "accepted" not in payload:
        payload["accepted"] = bool(payload["steps"])
    if reason_text:
        payload["reason"] = reason_text
    try:
        return PlanResult.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        # Provide detailed error message
        import traceback
        error_detail = str(exc)
        if hasattr(exc, "errors"):
            # Pydantic validation error
            error_detail = str(exc.errors())
        logger.error(f"Plan validation failed: {error_detail}\n{traceback.format_exc()}")
        logger.error(f"Payload: {payload}")
        logger.error(f"Steps: {payload.get('steps')}")
        return _reject(f"planner output did not match the test plan schema: {error_detail}")


class ChatModelPlanner:
    """Plans and writes scripts through the configured chat model."""

    def __init__(self, model: BaseChatModel, target_url: str) -> None:
        self.model = model
        self.target_url = target_url
        self.coding_instructions: list[dict] = []  # Track LLM's instructions to coding agent

    def plan(self, specification: str) -> PlanResult:
        # Prepend coding instructions context if available
        context = ""
        if self.coding_instructions:
            context = "\n\nContext from previous coding steps (what you instructed the coding agent to do):\n"
            for instr in self.coding_instructions:
                context += f"Step {instr.get('step')}:\n"
                context += f"  Action: {instr.get('action')}\n"
                if instr.get('coding_operations'):
                    for op in instr['coding_operations']:
                        context += f"  - {op.get('action')}"
                        if op.get('file_path'):
                            context += f": {op['file_path']}"
                        if op.get('content'):
                            # Show content preview
                            content = op['content']
                            if len(content) > 300:
                                content = content[:300] + "... (truncated)"
                            context += f"\n    Content: {content}"
                        if op.get('description'):
                            context += f"\n    Description: {op['description']}"
                        context += "\n"
                context += "\n"
            context += "IMPORTANT: When planning subsequent steps, be consistent with the instructions you gave above. "
            context += "For example, if you instructed the coding agent to create a service on port 5000, "
            context += "then your test steps should interact with port 5000, not a different port.\n\n"
        
        enhanced_specification = context + specification
        
        try:
            message = self.model.invoke(
                [
                    SystemMessage(content=_PLAN_SYSTEM),
                    HumanMessage(content=enhanced_specification),
                ]
            )
            content = message.content if isinstance(message.content, str) else str(message.content)
            result = _parse_plan(content)
        except Exception as exc:  # noqa: BLE001 - unparseable model output is a rejection
            import traceback
            detail = str(exc).splitlines()[0]
            logger.error(f"Plan parsing failed: {detail}\n{traceback.format_exc()}")
            return _reject(f"planner output did not match the test plan schema: {detail}")
        if not result.accepted and (
            _refused_to_plan(result.reason) or _structural_excuse(result.reason) or len(result.reason or "") < 160
        ):
            try:
                result = self._repair_plan(
                    enhanced_specification,
                    content,
                    (result.reason or "The draft rejected the request.")
                    + " You are the planner, not the executor. Write the phases instead. "
                    "A page check stays on the GUI phase that opens the page, "
                    "types into the named field, and clicks the named button. "
                    "Do not reject that sequence.",
                )
            except Exception as exc:  # noqa: BLE001 - a bad revision is a rejection
                detail = str(exc).splitlines()[0]
                return _reject(self._detail_rejection(specification, result.reason or detail))
        if not result.accepted:
            return _reject(self._detail_rejection(specification, result.reason))
        if result.phases:
            result = result.model_copy(update={"phases": fold_check_phases(result.phases)})
            gap = _validate_phases(result.phases)
            if gap or _looks_incomplete(result.reason):
                try:
                    result = self._repair_plan(specification, content, gap or result.reason or "")
                except Exception as exc:  # noqa: BLE001 - a bad revision is a rejection
                    detail = str(exc).splitlines()[0]
                    return _reject(f"planner output did not match the test plan schema: {detail}")
                if result.phases:
                    result = result.model_copy(update={"phases": fold_check_phases(result.phases)})
                if not result.accepted:
                    return _reject(self._detail_rejection(specification, result.reason))
        try:
            if result.phases:
                ordered, problem = order_phases(
                    omit_unrequested_stdout(
                        specification,
                        verbalize_verifications(specification, fold_check_phases(result.phases)),
                    )
                )
                if problem:
                    return _reject(self._detail_rejection(specification, problem))
                problem = _validate_phases(ordered)
                if problem:
                    return _reject(self._detail_rejection(specification, problem))
                steps = self._ensure_operations([_phase_to_step(phase) for phase in ordered])
            else:
                ordered = []
                steps = self._ensure_operations(_collapse_gui_sequences(result.steps))
        except Exception as exc:  # noqa: BLE001 - unparseable operation output is a rejection
            detail = str(exc).splitlines()[0]
            return _reject(f"planner output did not match the test plan schema: {detail}")
        problem = _validate_steps(steps) or _missing_operations(steps)
        if problem:
            return _reject(self._detail_rejection(specification, problem))
        return PlanResult(accepted=True, steps=steps, phases=ordered, reason=result.reason)

    def _repair_plan(self, specification: str, draft: str, finding: str) -> PlanResult:
        message = self.model.invoke(
            [
                SystemMessage(content=_REPAIR_SYSTEM),
                HumanMessage(
                    content=(
                        f"Testing request:\n{specification}\n\n"
                        f"Draft plan:\n{draft}\n\n"
                        f"What is wrong:\n{finding}"
                    )
                ),
            ]
        )
        revised = message.content if isinstance(message.content, str) else str(message.content)
        return _parse_plan(revised)

    def _detail_rejection(self, specification: str, finding: str | None) -> str:
        text = (finding or "").strip() or "The request could not be turned into a linear testing plan."
        if len(text) >= 160:
            return text
        try:
            message = self.model.invoke(
                [
                    SystemMessage(content=_REJECT_SYSTEM),
                    HumanMessage(
                        content=f"Testing request:\n{specification}\n\nWhy a plan was not produced:\n{text}"
                    ),
                ]
            )
            explained = message.content if isinstance(message.content, str) else str(message.content)
        except Exception:  # noqa: BLE001 - keep the finding when the explanation call fails
            return text
        explained = _strip_fence(explained).strip()
        if len(explained) < 80 or explained.startswith("{") or explained.startswith("["):
            return text
        return explained

    def _ensure_operations(self, steps: list[TestStep]) -> list[TestStep]:
        if not any(step.interface == "GUI" and not step.operations for step in steps):
            return steps
        lines = [f"Target URL: {self.target_url}"]
        for step in steps:
            if step.interface == "GUI" and not step.operations:
                lines.append(f"Step {step.step}: {step.phase_name or step.action}")
                if step.operation_notes:
                    lines.append("Operations: " + "; ".join(step.operation_notes))
                if step.verifications:
                    lines.append("Verifications: " + "; ".join(step.verifications))
        message = self.model.invoke(
            [
                SystemMessage(content=_OPS_SYSTEM),
                HumanMessage(content="\n".join(lines)),
            ]
        )
        content = message.content if isinstance(message.content, str) else str(message.content)
        by_step = _parse_operations(content)
        filled: list[TestStep] = []
        for step in steps:
            operations = by_step.get(step.step)
            if operations:
                actions = [GUIAction.model_validate(operation) for operation in operations]
                filled.append(step.model_copy(update={"operations": actions}))
            else:
                filled.append(step)
        return filled

    def script_for(self, intent: str) -> str:
        if "Assertion: json:" in intent or "json:" in intent:
            return (
                "import json\n"
                "from pathlib import Path\n"
                "files = sorted(Path('/evidence').glob('*.json'))\n"
                "print(files[0].read_text(encoding='utf-8') if files else '{}')\n"
            )
        command = command_from_intent(intent)
        if command:
            return command_script(command)
        message = self.model.invoke(
            [
                SystemMessage(content=_SCRIPT_SYSTEM),
                HumanMessage(content=intent),
            ]
        )
        content = message.content if isinstance(message.content, str) else str(message.content)
        return _strip_fence(content)

    def fix_step(self, step: dict, error: str, retry_count: int) -> dict:
        """Fix a failed step by asking the LLM for a corrected version."""
        try:
            import json
            step_text = json.dumps(step, indent=2, default=str)
            message = self.model.invoke(
                [
                    SystemMessage(content=_FIX_STEP_SYSTEM),
                    HumanMessage(
                        content=f"Failed step (attempt {retry_count + 1}):\n{step_text}\n\n"
                        f"Error:\n{error}\n\n"
                        f"Determine if this should be retried and provide a corrected version if so."
                    ),
                ]
            )
            content = message.content if isinstance(message.content, str) else str(message.content)
            content = _strip_fence(content)
            response = json.loads(content)

            # Check if LLM says this should not be retried
            if response.get("should_retry") is False:
                # Return the original step with a flag to indicate no retry
                step["_should_not_retry"] = True
                step["_retry_judgment"] = response.get("judgment", "Hard failure detected by LLM")
                return step
            
            # Get the fixed step from the response
            fixed_step = response.get("step", step)
            # Ensure the step number and interface are preserved
            fixed_step["step"] = step.get("step")
            fixed_step["interface"] = step.get("interface")
            return fixed_step
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to fix step: {exc}")
            return step  # Return original step if fixing fails

    def update_coding_context(self, coding_instructions: list[dict]) -> None:
        """Update the planner's context with coding instructions from previous steps."""
        self.coding_instructions = coding_instructions

    def _phases_from_model(self, raw_phases: list) -> tuple[list[TestPhase], list[TestStep], str | None]:
        phases, problem = _coerce_phases(raw_phases)
        if not problem:
            problem = _validate_phases(phases)
        if problem:
            return [], [], problem
        ordered, order_problem = order_phases(phases)
        if order_problem:
            return [], [], order_problem
        return ordered, [_phase_to_step(phase) for phase in ordered], None

    def refine_plan(self, current_phases: list[TestPhase], current_steps: list[TestStep],
                   user_feedback: str, chat_history: list[dict]) -> tuple[list[TestPhase], list[TestStep], str]:
        """Refine the plan based on user feedback. Returns (phases, steps, reasoning)."""
        import json
        
        # Build chat history context
        history_context = ""
        if chat_history:
            history_context = "\n\nChat history:\n"
            for msg in chat_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                history_context += f"{role}: {content}\n"
        
        # Build current plan context
        plan_context = "\n\nCurrent plan:\n"
        for phase in current_phases:
            plan_context += f"Phase {phase.phase}: {phase.name} ({phase.interface})\n"
            plan_context += f"  Depends on: {phase.depends_on}\n"
            plan_context += f"  Verifications: {phase.verifications}\n"
        for step in current_steps:
            plan_context += f"Step {step.step}: {step.action} ({step.interface})\n"
            plan_context += f"  Assertion: {step.assertion}\n"
            plan_context += f"  Verifications: {step.verifications}\n"
        
        try:
            message = self.model.invoke(
                [
                    SystemMessage(content=_REFINE_PLAN_SYSTEM),
                    HumanMessage(
                        content=f"User feedback: {user_feedback}\n\n{plan_context}\n\n{history_context}"
                    ),
                ]
            )
            content = message.content if isinstance(message.content, str) else str(message.content)
            content = _strip_fence(content)
            response = json.loads(content)
            
            # Check if this is an answer or a plan update
            if response.get("type") == "answer":
                # User asked a question, return the answer without changing the plan
                return current_phases, current_steps, response.get("answer", "Answer provided.")
            elif response.get("type") == "plan_update" or "phases" in response:
                phases, steps, problem = self._phases_from_model(response.get("phases") or [])
                if problem:
                    repaired = self.model.invoke(
                        [
                            SystemMessage(content=_REPAIR_SYSTEM),
                            HumanMessage(
                                content=(
                                    f"User feedback: {user_feedback}\n\n"
                                    f"Draft phases:\n{content}\n\n"
                                    f"What is wrong:\n{problem}"
                                )
                            ),
                        ]
                    )
                    repaired_text = repaired.content if isinstance(repaired.content, str) else str(repaired.content)
                    revised = _json_object(_strip_fence(repaired_text))
                    phases, steps, problem = self._phases_from_model(revised.get("phases") or [])
                if problem:
                    return current_phases, current_steps, f"The updated phases are not executable. {problem}"
                reasoning = response.get("reasoning") or "Plan refined based on user feedback."
                return phases, steps, reasoning
            else:
                return current_phases, current_steps, response.get("answer") or response.get("reasoning") or "Answer provided."
        except Exception as exc:  # noqa: BLE001
            import traceback
            logger.warning(f"Failed to refine plan: {exc}\n{traceback.format_exc()}")
            return current_phases, current_steps, f"Failed to refine plan: {exc}"


def _command_line(line: str) -> str:
    """Read a command out of a short operation such as 'run ls command'."""
    text = line.strip().strip("`").rstrip(".!?")
    text = re.sub(
        r"^(?:please\s+)?(?:run|execute|invoke)\s+(?:the\s+)?",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+command$", "", text, count=1, flags=re.IGNORECASE)
    return text.strip()


def command_from_intent(intent: str) -> str | None:
    """Return a shell command when the operation is one, rather than a prose step."""
    action = intent.split("\nAssertion:", 1)[0].strip()
    if action.lower().startswith("carry out:"):
        action = action.split(":", 1)[1].strip()
    raw = action.splitlines()[0].strip() if action else ""
    line = _command_line(raw)
    if not line:
        return None
    try:
        argv = shlex.split(line)
    except ValueError:
        return None
    if not argv:
        return None
    name = PurePath(argv[0]).name
    if not re.fullmatch(r"[A-Za-z0-9_./+-]+", argv[0]):
        return None
    if line.endswith((".", "?", "!")):
        return None
    lowered = f" {line.lower()} "
    if " the " in lowered and name.lower() not in _NETWORK_COMMANDS:
        return None
    return line


def command_needs_network(command: str) -> bool:
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    if not argv:
        return False
    return PurePath(argv[0]).name.lower() in _NETWORK_COMMANDS


def command_script(command: str) -> str:
    """Run the planned command and print its stdout."""
    argv = shlex.split(command)
    return (
        "import subprocess\n"
        f"completed = subprocess.run({argv!r}, capture_output=True, text=True, timeout=20)\n"
        "print(completed.stdout, end='')\n"
        "if completed.returncode:\n"
        "    raise SystemExit(completed.stderr or f'exit {completed.returncode}')\n"
    )


def build_planner(config: EngineConfig) -> Planner:
    from aqe.chat import get_chat_model

    return ChatModelPlanner(get_chat_model(), config.target_url)
