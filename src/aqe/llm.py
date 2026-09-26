"""Planners that turn a specification into a test matrix or a rejection."""

import json
import logging
import re
import shlex
from collections.abc import Callable
from pathlib import PurePath
from typing import Protocol, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from aqe.config import EngineConfig
from aqe.state import CodingAction, GUIAction, PlanResult, TestPhase, TestStep

logger = logging.getLogger(__name__)

T = TypeVar("T")


def invoke_json(model: BaseChatModel, messages: list, parser: Callable[[str], T]) -> tuple[T, str]:
    """Send one query and parse the reply as JSON."""
    message = model.invoke(messages)
    content = message.content if isinstance(message.content, str) else str(message.content)
    try:
        return parser(content), content
    except Exception as exc:
        logger.warning("Model reply was not valid JSON: %s", exc)
        raise ValueError(str(exc)) from exc

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
    "The agent can use three capabilities: a web browser (open a page, type into a field, click a button, press a key), "
    "host CLI tools (shell commands), and a Pi coding agent that generates code and files. "
    "The agent does not test desktop applications. "
    "A request to generate, write, or create source code or files, or a request for a CODING phase, "
    "is a CODING phase. The Pi coding agent runs it through coding_operations "
    "(create_file, update_file, or review_code), each with file_path, content, and description. "
    "Do not turn that request into a CLI command such as cat, tee, or python -c. "
    "The host CLI can start a process and call a network endpoint, including curl against a local health URL. "
    "A request to create an API or service and verify that it is running is accepted. "
    "Generate the source in a CODING phase. "
    "Start the process in the background and run the health check in a later CLI phase that depends on the CODING phase. "
    "Do not reject that request because the generated code must be executed, a server must be started, or a live endpoint must be called. "
    "If the request cannot be tested with the browser, CLI tools, or the Pi coding agent, "
    "set accepted to false and explain that mismatch in reason. "
)

_PLAN_SYSTEM = (
    _AGENT_CAPABILITIES
    + "You are the planner, not the executor. Another system will run the phases you write: "
    "GUI phases in the browser, CLI phases with host CLI tools, and CODING phases with the Pi coding agent. "
    "Do not reject a request because you cannot click, browse, read files, or write code yourself. "
    "Turn the testing request into one linear pipeline of testing phases. "
    "Reply with one JSON object only, with keys accepted, reason, and phases. "
    "Each phase is a chain of operations followed by the verifications of those operations. "
    "A phase has phase (an integer), name, depends_on (a list of earlier phase numbers, or empty), "
    "interface (GUI, CLI, or CODING), gui_driver (browser or null), operations, script, coding_operations, and verifications. "
    "GUI operations are objects with action goto, type, click, or press, plus text and selector {role, name} when needed. "
    "A CLI phase sets interface to CLI and puts the whole bash script in script. "
    "script is one string the agent runs with bash. Include every command, in order, with newlines escaped as \\n. "
    "Use set -e so a failing command stops the script. "
    "source, &&, and a trailing & belong in that script. Do not describe the commands in prose. "
    "CODING operations are objects with action create_file, update_file, or review_code, plus file_path, content, and description. "
    "When the user asks for a CODING phase, or asks the agent to generate code or files, that phase uses interface CODING "
    "and coding_operations. It does not use interface CLI. "
    "GUI verifications are questions about the page after the operations. "
    "A CLI verification is decided only from $? , stdout, and stderr after that phase's script runs. "
    "The script must print the evidence the check needs. "
    "Each CLI verification names what $? , stdout, or stderr must show, in one or two sentences. "
    "Do not write a vague check such as 'the background service is running', "
    "'the log is available for later inspection', 'a process is present', or 'the service is listening'. "
    "If a service is started in the background, the same script must then print proof, for example by calling curl on the health URL. "
    "The verification then says what that command produced, such as '$? is 0 and stdout is OK.' "
    "CODING verifications are questions about the files the Pi coding agent wrote. "
    "Name the file and what it should contain. "
    "The check is judged from the file that was written, not from a message that the operation succeeded. "
    "Write every verification as one or two complete sentences. Be specific and verbose. "
    "Keep the original meaning of the user's check. Do not add a condition they did not ask for, and do not drop one they did. "
    "Do not shorten a check into contains:, json:, or status:. "
    "If the user says a command returns nothing or mentions its output, say whether stdout is empty. "
    "Express a CLI check as a claim about $? , stdout, or stderr that the script prints. "
    "Mention stderr only when the check is about an error, issue, exception, warning, failure, traceback, or stderr. "
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
    + "Revise the testing plan so every phase is executable with the browser, a CLI command, or the Pi coding agent. "
    "Keep a requested CODING phase as interface CODING with coding_operations. "
    "Reply with one JSON object only, with keys accepted, reason, and phases. "
    "Use the same phase schema: phase, name, depends_on, interface, gui_driver, operations, script, coding_operations, and verifications. "
    "A CLI phase must include script, the complete bash script. "
    "A CLI verification is decided only from $? , stdout, and stderr. "
    "The script must print the evidence, and the verification must say what $? , stdout, or stderr must show. "
    "Do not write a vague check such as 'the background service is running' or 'the log is available for later inspection'. "
    "verifications must be strings. "
    "Every check the request asks for must appear as a verification written as one or two complete sentences. "
    "Be verbose and keep the original meaning. Do not shorten a check into contains:, json:, or status:. "
    "A command that should return nothing is described as empty stdout, not as an exit code. "
    "Express a CLI check as a claim about $? , stdout, or stderr that the script prints. "
    "Mention stderr only when the check is about an error, issue, exception, warning, failure, traceback, or stderr. "
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
    "Do not say the agent cannot start a server, execute generated code, or call a health endpoint. "
    "Those steps are a CLI phase after the CODING phase that writes the service. "
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
    "If the error is a hard failure, set should_retry to false and explain why in judgment, in one or two sentences. "
    "If the error can be fixed, set should_retry to true and provide the corrected step. "
    "The step is retried at most 3 times, and only when should_retry is true. "
    "Reply with one JSON object only. "
    "If should_retry is true, include the corrected step with these fields: step, interface, gui_driver, action, assertion, verifications, operations, script, coding_operations. "
    "Keep the same step number and interface. "
    "For a CLI step, script is the whole bash script to run. Replace that script to fix the failure. "
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
    + "You are a test planning assistant. The user may ask a question about the current plan or ask you to change it. "
    "Reply with one valid JSON object and nothing else. "
    "Do not add prose, markdown, or a code fence around it. "
    "Use double quotes for every key and string. "
    "Escape quotation marks and newlines inside strings. "
    "Do not use comments or trailing commas. "
    "If the user asks a question, return "
    '{"type": "answer", "answer": "the answer in sentences"}. '
    "If the user asks to change the plan, return the complete plan that should exist after the change: "
    '{"type": "plan_update", "reasoning": "one or two sentences", "phases": []}. '
    "Each phase has phase, name, interface, depends_on, operation_notes, script, verifications, and coding_operations. "
    "phase is a whole number: 1, then 2, then 3. Never use a decimal such as 1.5. "
    "To insert a phase, renumber every later phase. "
    "interface is GUI, CLI, or CODING. "
    "A GUI phase sets gui_driver to browser. "
    "A CLI phase sets script to the complete bash script, with newlines escaped as \\n. "
    "Do not put the commands in prose. "
    "A CLI verification names what $? , stdout, or stderr must show after the script runs. "
    "The script must print that evidence. "
    "Do not write a vague check such as 'the background service is running' or 'the log is available for later inspection'. "
    "Each CLI script starts a new shell. source does not carry into the next phase. "
    "A later phase that needs the virtual environment must source it again at the start of its script, "
    "or call venv/bin/pip and venv/bin/python. "
    "The phases array is the complete plan after the change. "
    "The agent stores that list as the plan. "
    "Include every phase that should still run, with its script or its file content. "
    "When the user asks to remove or merge phases, leave those phases out of the list and fix depends_on so every dependency is a phase in the list. "
    "A CODING phase puts create_file, update_file, or review_code objects in coding_operations, "
    "each with action, file_path, content, and description."
)

_VALID_JSON_SYSTEM = (
    "Reply with one valid JSON object and nothing else. "
    "Use double quotes. Escape quotation marks and newlines inside strings. "
    "Do not use comments, trailing commas, or a markdown fence."
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
    "cannot be executed",
    "cannot construct",
    "as an llm",
    "i cannot",
    "i do not",
    "cannot browse",
    "cannot access",
    "start a server",
    "network endpoint",
    "live network",
    "health check",
    "available tools",
)

_SERVICE_PIPELINE = (
    "Host CLI tools start a process and call a network endpoint. "
    "Generate the service source in a CODING phase with coding_operations. "
    "Start that process in the background, then in that same CLI phase or a later CLI phase print proof with a concrete command such as curl. "
    "The verification names $? and the stdout of that command. "
    "Use a concrete port. "
    "Do not reject the request because the generated code must run or a live endpoint must be called. "
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


_CODING_ACTION_ALIASES = {
    "write_file": "create_file",
    "create_file": "create_file",
    "update_file": "update_file",
    "edit_file": "update_file",
    "modify_file": "update_file",
    "review_code": "review_code",
    "execute_code": "execute_code",
    "execute_command": "execute_code",
    "execute_shell": "execute_code",
    "run_command": "execute_code",
    "run_code": "execute_code",
    "make_executable": "execute_code",
    "run": "execute_code",
}


def _coding_action_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return _CODING_ACTION_ALIASES.get(value.strip().lower())


def _normalize_coding_operation(raw: dict) -> dict | None:
    """Read a coding operation when the model uses a nearby field name or nests the action."""
    op = dict(raw)
    if "action" not in op:
        nested = [
            key
            for key, value in op.items()
            if _coding_action_name(key) and isinstance(value, dict)
        ]
        if len(nested) == 1 and len(op) == 1:
            key = nested[0]
            op = dict(op[key])
            op["action"] = _coding_action_name(key)
        else:
            for alias in ("coding_operation", "coding_action", "operation", "type", "op"):
                if alias in op:
                    op["action"] = op.pop(alias)
                    break
    if "command" in op and "action" not in op:
        op["action"] = "execute_code"
        op["content"] = op.pop("command")
    if not op.get("file_path"):
        for key in ("path", "file", "filename", "filepath"):
            if isinstance(op.get(key), str) and op[key].strip():
                op["file_path"] = op.pop(key)
                break
    if op.get("content") is None:
        for key in ("code", "text", "body", "source"):
            if isinstance(op.get(key), str):
                op["content"] = op.pop(key)
                break
    action = _coding_action_name(op.get("action")) or _infer_coding_action(op)
    if action is None:
        return None
    op["action"] = action
    return op


def _infer_coding_action(op: dict) -> str | None:
    """A file and its content are a create when the model omits the action name."""
    description = str(op.get("description") or "")
    has_content = isinstance(op.get("content"), str)
    has_path = isinstance(op.get("file_path"), str) and bool(op["file_path"].strip())
    if re.search(r"\breview\b", description, re.IGNORECASE) and not has_content:
        return "review_code"
    if re.search(r"\b(update|edit|modify)\b", description, re.IGNORECASE) and has_path:
        return "update_file"
    if has_content or re.search(r"\b(create|write|add)\b", description, re.IGNORECASE):
        return "create_file"
    if has_path:
        return "review_code"
    return None


def _coding_operations_from(raw: object) -> list[CodingAction]:
    items = [raw] if isinstance(raw, dict) else raw if isinstance(raw, list) else []
    operations: list[CodingAction] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_coding_operation(item)
        if normalized is None:
            logger.warning(f"Failed to validate coding operation: {item}")
            continue
        try:
            operations.append(CodingAction.model_validate(normalized))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to validate coding operation: {exc}")
    return operations


def _bash_script(raw: object, notes: list[str]) -> str:
    """The bash program for a CLI phase. A missing script is built from command notes."""
    if isinstance(raw, str) and raw.strip():
        text = _strip_fence(raw).strip()
        return text if text.endswith("\n") else f"{text}\n"
    command = command_from_intent("\n".join(notes))
    if not command:
        return ""
    return f"set -e\n{command}\n"


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
        if not notes:
            notes = _string_list(item.get("operation_notes"))
        verifications = _string_list(item.get("verifications") or item.get("verification") or item.get("assertions"))
        if not name:
            name = notes[0] if notes else f"Phase {number}"
        interface = str(item.get("interface") or "").strip().upper()
        driver = item.get("gui_driver")
        driver_name = str(driver).strip().lower() if isinstance(driver, str) and driver.strip() else None

        coding_operations = _coding_operations_from(item.get("coding_operations") or [])
        if not coding_operations:
            coding_operations = _coding_operations_from(item.get("operations") or [])

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
        script = _bash_script(
            item.get("script") or item.get("bash") or item.get("bash_script") or item.get("shell"),
            notes,
        )
        if interface == "CLI" and verifications and not notes and not actions and not script.strip():
            notes = [f"Carry out: {item}" for item in verifications]
        if interface == "CODING" and not verifications and notes:
            verifications = list(notes)
        if interface == "CODING" and verifications and not notes and not coding_operations:
            notes = [f"Carry out: {item}" for item in verifications]
        if interface == "CLI" and not script.strip():
            script = _bash_script("", notes)

        phases.append(
            TestPhase(
                phase=number,
                name=name,
                interface=interface,  # type: ignore[arg-type]
                gui_driver=driver_name,  # type: ignore[arg-type]
                depends_on=_depends_on(item.get("depends_on") or item.get("after")),
                operations=[GUIAction.model_validate(action) for action in actions],
                operation_notes=notes,
                script=script if interface == "CLI" else "",
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
    return bool(phase.script.strip() or phase.operations or phase.operation_notes or phase.coding_operations)


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


_CODING_REQUEST = re.compile(
    r"\bCODING\b|coding phase|coding agent|pi coding|pi agent|"
    r"generate (?:the )?(?:code|files?)|write (?:the )?code",
    re.IGNORECASE,
)


def _missing_coding_phase(text: str, phases: list[TestPhase]) -> str | None:
    """A request for the Pi coding agent must stay a CODING phase."""
    if _CODING_REQUEST.search(text) is None:
        return None
    if any(phase.interface == "CODING" and phase.coding_operations for phase in phases):
        return None
    return (
        "The request asks for a CODING phase. The Pi coding agent generates the code or files. "
        "Use interface CODING and coding_operations (create_file, update_file, or review_code). "
        "Do not replace that phase with a CLI command."
    )


def _apply_phase_update(
    current: list[TestPhase],
    raw_phases: list,
    feedback: str,
) -> tuple[list[TestPhase], list[TestStep], str | None]:
    """Store the phase list the model returned. The model decides which phases remain."""
    del current
    phases, problem = _coerce_phases(_integer_phases(raw_phases))
    if problem:
        return [], [], problem
    if not phases:
        return [], [], "The update did not include any phases."
    problem = _validate_phases(phases) or _missing_coding_phase(feedback, phases)
    if problem:
        return [], [], problem
    ordered, order_problem = order_phases(phases)
    if order_problem:
        return [], [], order_problem
    return ordered, [_phase_to_step(phase) for phase in ordered], None


def _integer_phases(raw: list) -> list:
    """Give every phase a whole number, in the order the model listed them."""
    if not raw or not all(isinstance(item, dict) for item in raw):
        return raw
    labels = [item.get("phase", item.get("id", index)) for index, item in enumerate(raw, start=1)]
    numbers: list[int] = []
    for label in labels:
        if isinstance(label, bool) or isinstance(label, float) or not isinstance(label, (int, str)):
            numbers = []
            break
        if isinstance(label, str) and not label.isdigit():
            numbers = []
            break
        numbers.append(int(label))
    if numbers and len(set(numbers)) == len(numbers):
        return raw
    mapping = {str(label): index for index, label in enumerate(labels, start=1)}
    renumbered: list[dict] = []
    for index, item in enumerate(raw, start=1):
        updated = dict(item)
        updated["phase"] = index
        depends = updated.get("depends_on", updated.get("after", []))
        if isinstance(depends, list):
            updated["depends_on"] = [
                mapping[str(dep)]
                for dep in depends
                if str(dep) in mapping and mapping[str(dep)] != index
            ]
        renumbered.append(updated)
    return renumbered


def _validate_phases(phases: list[TestPhase]) -> str | None:
    if not phases:
        return (
            "The plan has no testing phases. A request needs a pipeline of phases, "
            "and each phase needs operations to carry out."
        )
    problems: list[str] = []
    for phase in phases:
        label = f"Phase {phase.phase} ({phase.name})"
        has_operations = bool(
            phase.script.strip() or phase.operation_notes or phase.operations or phase.coding_operations
        )
        if not has_operations:
            problems.append(
                f"{label} has no operations. "
                f"A phase has to carry out a chain of operations to be actionable."
            )
        if phase.interface == "CODING" and not phase.coding_operations:
            problems.append(
                f"{label} is a CODING phase with no coding operations. "
                "Include coding_operations for the Pi coding agent: create_file, update_file, or review_code."
            )
        if phase.interface == "GUI" and phase.gui_driver == "desktop":
            problems.append(
                f"{label} targets a desktop application. Desktop applications are not tested. "
                "Use the browser, a CLI command, or a CODING phase."
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
        script=phase.script,
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


def _string_ends_here(text: str, quote_index: int) -> bool:
    """A quote ends a JSON string when the next token is a comma or a closing brace."""
    rest = text[quote_index + 1 :]
    index = 0
    while index < len(rest) and rest[index] in " \t\r\n":
        index += 1
    if index >= len(rest):
        return True
    char = rest[index]
    if char == ",":
        return True
    if char in "}]":
        index += 1
        while index < len(rest) and rest[index] in " \t\r\n":
            index += 1
        return index >= len(rest) or rest[index] in ",}]"
    return False


def _key_ends_here(text: str, quote_index: int) -> bool:
    """A quote ends a JSON key when the next token is a colon."""
    rest = text[quote_index + 1 :]
    index = 0
    while index < len(rest) and rest[index] in " \t\r\n":
        index += 1
    return index >= len(rest) or rest[index] == ":"


def _escape_interior_quotes(text: str) -> str:
    """Escape quotes and line breaks that sit inside a JSON string, such as source code."""
    out: list[str] = []
    index = 0
    length = len(text)
    in_string = False
    escape = False
    role = "value"
    stack: list[str] = []
    expect = "value"

    while index < length:
        char = text[index]
        if in_string:
            if escape:
                escape = False
                out.append(char)
                index += 1
                continue
            if char == "\\":
                escape = True
                out.append(char)
                index += 1
                continue
            if char == '"':
                ends = _key_ends_here(text, index) if role == "key" else _string_ends_here(text, index)
                if ends:
                    in_string = False
                    expect = "colon" if role == "key" else "comma"
                    out.append(char)
                else:
                    out.append('\\"')
                index += 1
                continue
            if char == "\n":
                out.append("\\n")
                index += 1
                continue
            if char == "\r":
                out.append("\\r")
                index += 1
                continue
            if char == "\t":
                out.append("\\t")
                index += 1
                continue
            out.append(char)
            index += 1
            continue
        if char.isspace():
            out.append(char)
            index += 1
            continue
        if char == '"':
            role = "key" if expect == "key" else "value"
            in_string = True
            out.append(char)
            index += 1
            continue
        if char in "{[":
            stack.append(char)
            expect = "key" if char == "{" else "value"
            out.append(char)
            index += 1
            continue
        if char in "}]":
            if stack:
                stack.pop()
            expect = "comma"
            out.append(char)
            index += 1
            continue
        if char == ":":
            expect = "value"
            out.append(char)
            index += 1
            continue
        if char == ",":
            expect = "key" if stack and stack[-1] == "{" else "value"
            out.append(char)
            index += 1
            continue
        if char == "-" or char.isdigit() or text.startswith(("true", "false", "null"), index):
            if text.startswith("true", index):
                token = "true"
            elif text.startswith("false", index):
                token = "false"
            elif text.startswith("null", index):
                token = "null"
            else:
                end = index + 1
                while end < length and text[end] in "0123456789.eE+-":
                    end += 1
                token = text[index:end]
            out.append(token)
            index += len(token)
            expect = "comma"
            continue
        out.append(char)
        index += 1
    if in_string:
        out.append('"')
    return "".join(out)


def _escape_string_controls(text: str) -> str:
    """Turn raw line breaks inside JSON strings into escaped characters."""
    out: list[str] = []
    in_string = False
    escape = False
    for char in text:
        if in_string:
            if escape:
                escape = False
                out.append(char)
                continue
            if char == "\\":
                escape = True
                out.append(char)
                continue
            if char == '"':
                in_string = False
                out.append(char)
                continue
            if char == "\n":
                out.append("\\n")
                continue
            if char == "\r":
                out.append("\\r")
                continue
            if char == "\t":
                out.append("\\t")
                continue
        elif char == '"':
            in_string = True
        out.append(char)
    return "".join(out)


def _insert_missing_commas(text: str) -> str:
    """Insert a comma when a JSON value is followed by another value."""
    out: list[str] = []
    index = 0
    length = len(text)
    in_string = False
    escape = False
    stack: list[str] = []
    expect = "value"

    def value_start(char: str) -> bool:
        return char in '{["' or char.isdigit() or char in "-tfn"

    while index < length:
        char = text[index]
        if in_string:
            out.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
                expect = "comma"
            index += 1
            continue
        if char.isspace():
            out.append(char)
            index += 1
            continue
        if expect == "comma" and char not in ",}]" and value_start(char):
            out.append(",")
            expect = "key" if stack and stack[-1] == "{" else "value"
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char in "{[":
            stack.append(char)
            expect = "key" if char == "{" else "value"
            out.append(char)
            index += 1
            continue
        if char in "}]":
            if stack:
                stack.pop()
            expect = "comma"
            out.append(char)
            index += 1
            continue
        if char == ":":
            expect = "value"
            out.append(char)
            index += 1
            continue
        if char == ",":
            expect = "key" if stack and stack[-1] == "{" else "value"
            out.append(char)
            index += 1
            continue
        if char == "-" or char.isdigit() or text.startswith(("true", "false", "null"), index):
            if text.startswith("true", index):
                token = "true"
            elif text.startswith("false", index):
                token = "false"
            elif text.startswith("null", index):
                token = "null"
            else:
                end = index + 1
                while end < length and text[end] in "0123456789.eE+-":
                    end += 1
                token = text[index:end]
            out.append(token)
            index += len(token)
            expect = "comma"
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _fix_bracket_mismatch(text: str) -> str:
    """Use the bracket that matches the open container when the model swaps ] and }."""
    out: list[str] = []
    index = 0
    length = len(text)
    in_string = False
    escape = False
    stack: list[str] = []
    while index < length:
        char = text[index]
        if in_string:
            out.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char in "{[":
            stack.append(char)
            out.append(char)
            index += 1
            continue
        if char == "}" and stack and stack[-1] == "[":
            stack.pop()
            out.append("]")
            index += 1
            continue
        if char == "]" and stack and stack[-1] == "{":
            stack.pop()
            out.append("}")
            index += 1
            continue
        if char in "}]":
            if stack:
                stack.pop()
            out.append(char)
            index += 1
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _json_object(content: str) -> dict:
    text = _strip_fence(content)
    start = text.find("{")
    if start < 0:
        raise ValueError("planner output did not match the test plan schema")
    blob = text[start:]
    escaped = _escape_string_controls(blob)
    quoted = _escape_interior_quotes(blob)
    quoted_escaped = _escape_interior_quotes(escaped)
    bracketed = _fix_bracket_mismatch(blob)
    candidates = (
        blob,
        escaped,
        bracketed,
        _insert_missing_commas(blob),
        _insert_missing_commas(escaped),
        _insert_missing_commas(bracketed),
        quoted,
        quoted_escaped,
        _insert_missing_commas(quoted),
        _insert_missing_commas(quoted_escaped),
        _fix_bracket_mismatch(quoted),
        _fix_bracket_mismatch(quoted_escaped),
        _insert_missing_commas(_fix_bracket_mismatch(quoted)),
        _insert_missing_commas(_fix_bracket_mismatch(quoted_escaped)),
        _repair_json(blob),
        _insert_missing_commas(_repair_json(escaped)),
    )
    last_error = "no JSON object found"
    for candidate in candidates:
        try:
            value, _ = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError as exc:
            last_error = f"{exc.msg} (line {exc.lineno} column {exc.colno})"
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(last_error)


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
        phases, problem = _coerce_phases(_integer_phases(raw_phases))
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
        
        messages = [
            SystemMessage(content=_PLAN_SYSTEM),
            HumanMessage(content=enhanced_specification),
        ]
        try:
            result, content = invoke_json(self.model, messages, _parse_plan)
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
                    + _SERVICE_PIPELINE
                    + "A page check stays on the GUI phase that opens the page, "
                    "types into the named field, and clicks the named button. "
                    "Do not reject that sequence.",
                )
            except Exception as exc:  # noqa: BLE001 - a bad revision is a rejection
                detail = str(exc).splitlines()[0]
                return _reject(self._detail_rejection(specification, result.reason or detail))
            if not result.accepted and _refused_to_plan(result.reason):
                try:
                    result = self._repair_plan(
                        enhanced_specification,
                        content,
                        (result.reason or "The draft rejected the request.")
                        + " "
                        + _SERVICE_PIPELINE
                        + "Set accepted to true and return those phases.",
                    )
                except Exception as exc:  # noqa: BLE001 - a bad revision is a rejection
                    detail = str(exc).splitlines()[0]
                    return _reject(self._detail_rejection(specification, result.reason or detail))
        if not result.accepted:
            return _reject(self._detail_rejection(specification, result.reason))
        if result.phases:
            result = result.model_copy(update={"phases": fold_check_phases(result.phases)})
            gap = _validate_phases(result.phases) or _missing_coding_phase(specification, result.phases)
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
                problem = _validate_phases(ordered) or _missing_coding_phase(specification, ordered)
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
        messages = [
            SystemMessage(content=_REPAIR_SYSTEM),
            HumanMessage(
                content=(
                    f"Testing request:\n{specification}\n\n"
                    f"Draft plan:\n{draft}\n\n"
                    f"What is wrong:\n{finding}"
                )
            ),
        ]
        revised, _content = invoke_json(self.model, messages, _parse_plan)
        return revised

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
        messages = [
            SystemMessage(content=_OPS_SYSTEM),
            HumanMessage(content="\n".join(lines)),
        ]
        by_step, _content = invoke_json(self.model, messages, _parse_operations)
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
            messages = [
                SystemMessage(content=_FIX_STEP_SYSTEM),
                HumanMessage(
                    content=f"Failed step (attempt {retry_count + 1}):\n{step_text}\n\n"
                    f"Error:\n{error}\n\n"
                    f"Determine if this should be retried and provide a corrected version if so."
                ),
            ]
            response, _content = invoke_json(self.model, messages, _json_object)
            retry = response.get("should_retry")
            approved = retry is True or (isinstance(retry, str) and retry.strip().lower() == "true")
            if not approved:
                step["_should_not_retry"] = True
                step["_retry_judgment"] = response.get("judgment") or "The model classified this as a hard failure."
                return step
            proposed = response.get("step") if isinstance(response.get("step"), dict) else {}
            fixed_step = dict(step)
            fixed_step.update({key: value for key, value in proposed.items() if value is not None})
            fixed_step.pop("_should_not_retry", None)
            fixed_step.pop("_retry_judgment", None)
            fixed_step["step"] = step.get("step")
            fixed_step["interface"] = step.get("interface")
            return fixed_step
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to fix step: {exc}")
            step["_should_not_retry"] = True
            step["_retry_judgment"] = "The model did not return a usable step update."
            return step

    def update_coding_context(self, coding_instructions: list[dict]) -> None:
        """Update the planner's context with coding instructions from previous steps."""
        self.coding_instructions = coding_instructions

    def refine_plan(self, current_phases: list[TestPhase], current_steps: list[TestStep],
                   user_feedback: str, chat_history: list[dict]) -> tuple[list[TestPhase], list[TestStep], str]:
        """Refine the plan based on user feedback. Returns (phases, steps, reasoning)."""
        # Build chat history context
        history_context = ""
        if chat_history:
            history_context = "\n\nChat history:\n"
            for msg in chat_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                history_context += f"{role}: {content}\n"
        
        plan_json = json.dumps([phase.model_dump() for phase in current_phases], indent=2)
        messages = [
            SystemMessage(content=_REFINE_PLAN_SYSTEM),
            HumanMessage(
                content=(
                    f"User feedback: {user_feedback}\n\n"
                    f"Current phases JSON:\n{plan_json}\n\n"
                    f"{history_context}"
                )
            ),
        ]
        try:
            response, content = invoke_json(self.model, messages, _json_object)
            
            # Check if this is an answer or a plan update
            if response.get("type") == "answer":
                # User asked a question, return the answer without changing the plan
                return current_phases, current_steps, response.get("answer", "Answer provided.")
            elif response.get("type") == "plan_update" or "phases" in response:
                phases, steps, problem = _apply_phase_update(
                    current_phases,
                    response.get("phases") or [],
                    user_feedback,
                )
                if problem:
                    repair_messages = [
                        SystemMessage(content=_REPAIR_SYSTEM),
                        HumanMessage(
                            content=(
                                f"User feedback: {user_feedback}\n\n"
                                f"Current phases must all remain unless the user asked to remove one.\n"
                                f"Draft phases:\n{content}\n\n"
                                f"What is wrong:\n{problem}"
                            )
                        ),
                    ]
                    revised, _repaired_text = invoke_json(self.model, repair_messages, _json_object)
                    phases, steps, problem = _apply_phase_update(
                        current_phases,
                        revised.get("phases") or [],
                        user_feedback,
                    )
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


def _looks_like_command(text: str) -> bool:
    """A shell command starts with a program name, not a prose label."""
    try:
        argv = shlex.split(text)
    except ValueError:
        return False
    if not argv:
        return False
    token = argv[0]
    if token == "source":
        return True
    if not re.fullmatch(r"[A-Za-z0-9_./+-]+", token):
        return False
    if any(char.isupper() for char in token) and "/" not in token:
        return False
    lowered = f" {text.lower()} "
    if " the " in lowered and PurePath(token).name.lower() not in _NETWORK_COMMANDS:
        return False
    return True


def _command_from_line(line: str) -> str | None:
    text = re.sub(r"^\d+[\).]\s*", "", line.strip())
    text = _command_line(text)
    if not text:
        return None
    if _looks_like_command(text):
        return text
    for match in re.finditer(r":\s*(\S.*?)\s*$", text):
        tail = match.group(1).strip().rstrip(".!?")
        if _looks_like_command(tail):
            return tail
    return None


def command_from_intent(intent: str) -> str | None:
    """Return a shell command when the operation is one, rather than a prose step."""
    action = intent.split("\nAssertion:", 1)[0].strip()
    if action.lower().startswith("carry out:"):
        action = action.split(":", 1)[1].strip()
    pieces: list[str] = []
    for line in action.splitlines():
        pieces.extend(part.strip() for part in re.split(r"(?<=\.)\s+(?=[A-Z])", line) if part.strip())
    commands: list[str] = []
    for piece in pieces:
        command = _command_from_line(piece)
        if command and command not in commands:
            commands.append(command)
    if not commands:
        return None
    if len(commands) == 1:
        return commands[0]
    if any(command.rstrip().endswith("&") for command in commands):
        return "\n".join(commands)
    return " && ".join(commands)


def command_needs_network(command: str) -> bool:
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    names = {PurePath(token).name.lower() for token in argv}
    return bool(names & {name.lower() for name in _NETWORK_COMMANDS})


def _needs_shell(command: str) -> bool:
    return bool(re.search(r"&&|\|\||[;|&<>]|`|\$\(|\bsource\b|\n", command))


def command_script(command: str) -> str:
    """Run the planned command and print its stdout."""
    if _needs_shell(command):
        launched = f"subprocess.run({command!r}, shell=True, executable='/bin/bash', capture_output=True, text=True, timeout=120)"
    else:
        argv = shlex.split(command)
        launched = f"subprocess.run({argv!r}, capture_output=True, text=True, timeout=20)"
    return (
        "import subprocess\n"
        f"completed = {launched}\n"
        "print(completed.stdout, end='')\n"
        "if completed.returncode:\n"
        "    raise SystemExit(completed.stderr or f'exit {completed.returncode}')\n"
    )


def build_planner(config: EngineConfig) -> Planner:
    from aqe.chat import get_chat_model

    return ChatModelPlanner(get_chat_model(), config.target_url)
