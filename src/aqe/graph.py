"""Plan, preflight, route, execute, validate, and reflect."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from aqe.assertions import evaluate_assertion
from aqe.capabilities import (
    Probe,
    missing_details,
    probe_host,
    required_capabilities,
)
from aqe.cli_runtime.synthesizer import CLISubsystem
from aqe.coding_agent.subsystem import CodingSubsystem, PiAgentError
from aqe.config import EngineConfig
from aqe.state import PlanResult
from aqe.errors import HarnessError
from aqe.gui.subsystem import GUISubsystem
from aqe.judge import PageJudge, build_judge, judge_stderr
from aqe.llm import Planner
from aqe.state import (
    ActionResult,
    AgentState,
    Judgment,
    StepView,
    TestReport,
    TestStep,
    VerificationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class RunControl:
    cancel_requested: bool = False


@dataclass
class GraphDeps:
    config: EngineConfig
    planner: Planner
    gui: GUISubsystem
    cli: CLISubsystem
    coding: CodingSubsystem
    control: RunControl
    probe: Probe
    evidence_dir_for: Callable[[str], str]
    judge: PageJudge | None = None
    chat_model: Any = None
    run_dir: Path | None = None
    work_dir: Path | None = None


def _views(state: AgentState) -> list[StepView]:
    return [StepView.model_validate(item) for item in state.get("step_views", [])]


def _store_views(views: list[StepView]) -> list[dict]:
    return [view.model_dump() for view in views]


def _matrix(state: AgentState) -> list[TestStep]:
    matrix_data = state.get("test_matrix", [])
    matrix: list[TestStep] = []
    for item in matrix_data:
        try:
            # Coerce operations field before validation
            if isinstance(item, dict):
                interface = item.get("interface", "").upper()
                operations = item.get("operations")
                
                # For CODING steps, operations should be empty
                if interface == "CODING" and operations:
                    logger.info(f"Clearing operations for CODING step {item.get('step')}")
                    item["operations"] = []
                
                # For CLI steps, handle operations that contain CLI commands
                if interface == "CLI" and isinstance(operations, list):
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
                            item["action"] = " && ".join(cli_commands)
                        # For CLI steps, operations should be empty or only GUI actions
                        item["operations"] = gui_actions if gui_actions else []
                
                # Coerce coding operations
                coding_operations = item.get("coding_operations", [])
                if isinstance(coding_operations, list):
                    for op in coding_operations:
                        if isinstance(op, dict):
                            # Fix common LLM mistakes: rename 'type' to 'action'
                            if "type" in op and "action" not in op:
                                op["action"] = op.pop("type")
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
                                    "execute_shell": "execute_code",
                                    "run_command": "execute_code",
                                    "run": "execute_code",
                                    "run_code": "execute_code",
                                    "make_executable": "execute_code",
                                }
                                if op["action"] in type_to_action:
                                    op["action"] = type_to_action[op["action"]]
                                else:
                                    # Unknown action - default to execute_code for coding operations
                                    logger.warning(f"Unknown coding action '{op['action']}', defaulting to execute_code")
                                    op["action"] = "execute_code"
                            # Fix common LLM mistakes: rename 'operation' to 'action'
                            if "operation" in op and "action" not in op:
                                op["action"] = op.pop("operation")
                            # Fix common LLM mistakes: rename 'execute_command' to 'execute_code'
                            if op.get("action") == "execute_command":
                                op["action"] = "execute_code"
                            # Fix common LLM mistakes: rename 'execute_shell' to 'execute_code'
                            if op.get("action") == "execute_shell":
                                op["action"] = "execute_code"
                            # Fix common LLM mistakes: rename 'run_command' to 'execute_code'
                            if op.get("action") == "run_command":
                                op["action"] = "execute_code"
                            # Fix common LLM mistakes: rename 'run' to 'execute_code' for coding operations
                            if op.get("action") == "run":
                                op["action"] = "execute_code"
            
            matrix.append(TestStep.model_validate(item))
        except Exception as exc:  # noqa: BLE001
            import traceback
            logger.error(f"Failed to validate TestStep: {exc}\n{traceback.format_exc()}")
            logger.error(f"Item data: {item}")
            raise
    return matrix


def _set_status(views: list[StepView], index: int, status: str, **updates: object) -> None:
    view = views[index].model_copy(update={"status": status, **updates})
    views[index] = view


def plan_node(state: AgentState, deps: GraphDeps) -> dict:
    try:
        # Check if there's an approved plan in storage
        from aqe.plan_storage import PlanStorageManager
        plan_storage = PlanStorageManager(deps.config.runs_dir)
        plan = plan_storage.load_plan(state["run_id"])
        
        if plan and plan.status == "approved":
            # Use the approved plan from storage
            result = PlanResult(accepted=True, phases=plan.phases, steps=plan.steps)
        else:
            # Generate a new plan
            result = deps.planner.plan(state["specification"])
    except Exception as exc:  # noqa: BLE001 - bad planner output rejects the run
        import traceback
        logger.error(f"Planner failed: {exc}\n{traceback.format_exc()}")
        return {
            "phase": "finish",
            "reason_code": "not_a_test_plan",
            "reason": f"planner output did not match the test plan schema: {exc}",
            "step_views": [],
            "test_matrix": [],
        }
    if not result.accepted:
        return {
            "phase": "finish",
            "reason_code": result.reason_code or "not_a_test_plan",
            "reason": result.reason or "This request has no verifiable GUI or CLI actions.",
            "step_views": [],
            "test_matrix": [],
        }
    problem = None
    for step in result.steps:
        problem = step.validate_shape()
        if problem:
            break
    if problem or not result.steps:
        return {
            "phase": "finish",
            "reason_code": "not_a_test_plan",
            "reason": problem or "no verifiable GUI or CLI actions were found",
            "step_views": [],
            "test_matrix": [],
        }
    views = [StepView.from_step(step) for step in result.steps]
    return {
        "phase": "preflight",
        "test_matrix": [step.model_dump() for step in result.steps],
        "step_views": _store_views(views),
        "current_step": 0,
        "reason_code": None,
        "reason": None,
    }


def preflight_node(state: AgentState, deps: GraphDeps) -> dict:
    capabilities = deps.probe()
    print(f"[DEBUG] Preflight: capabilities = {capabilities}")
    needed = required_capabilities(
        state.get("test_matrix", []),
        llm=deps.config.llm,
        llm_configured=deps.config.llm_configured,
    )
    print(f"[DEBUG] Preflight: needed capabilities = {needed}")
    missing = []
    flags = {
        "browser": capabilities.browser.available,
        "desktop": capabilities.desktop.available,
        "coding": capabilities.coding.available,
        "llm": deps.config.llm_configured,
    }
    print(f"[DEBUG] Preflight: flags = {flags}")
    for name in needed:
        if name == "llm":
            if not deps.config.llm_configured:
                missing.append(name)
            continue
        if not flags[name]:
            missing.append(name)
    print(f"[DEBUG] Preflight: missing = {missing}")
    if missing:
        details = missing_details(capabilities, missing)
        return {
            "phase": "finish",
            "reason_code": "missing_capability",
            "reason": " ".join(details) or "A required driver is unavailable.",
            "missing": missing,
        }

    return {"phase": "route", "missing": []}


def _error_update(state: AgentState, index: int, code: str, message: str) -> dict:
    views = _views(state)
    _set_status(views, index, "error", assertion_passed=None, summary=message)
    for later in range(index + 1, len(views)):
        _set_status(views, later, "skipped", assertion_passed=None)
    return {
        "phase": "finish",
        "reason_code": code,
        "reason": message,
        "step_views": _store_views(views),
    }


def _error_update(state: AgentState, index: int, code: str, message: str) -> dict:
    views = _views(state)
    _set_status(views, index, "error", assertion_passed=None, summary=message)
    for later in range(index + 1, len(views)):
        _set_status(views, later, "skipped", assertion_passed=None)
    return {
        "phase": "finish",
        "reason_code": code,
        "reason": message,
        "step_views": _store_views(views),
    }


def _error_update(state: AgentState, index: int, code: str, message: str) -> dict:
    views = _views(state)
    _set_status(views, index, "error", assertion_passed=None, summary=message)
    for later in range(index + 1, len(views)):
        _set_status(views, later, "skipped", assertion_passed=None)
    return {
        "phase": "finish",
        "reason_code": code,
        "reason": message,
        "step_views": _store_views(views),
    }


def _error_update(state: AgentState, index: int, code: str, message: str) -> dict:
    views = _views(state)
    _set_status(views, index, "error", assertion_passed=None, summary=message)
    for later in range(index + 1, len(views)):
        _set_status(views, later, "skipped", assertion_passed=None)
    return {
        "phase": "finish",
        "reason_code": code,
        "reason": message,
        "step_views": _store_views(views),
    }


def route_node(state: AgentState, deps: GraphDeps) -> dict:
    print(f"[DEBUG] Route node: cancel_requested={deps.control.cancel_requested}")
    if deps.control.cancel_requested:
        return {"phase": "finish", "reason_code": "canceled", "reason": "The run was canceled."}
    matrix = _matrix(state)
    index = state.get("current_step", 0)
    print(f"[DEBUG] Route node: current_step={index}, matrix length={len(matrix)}")
    if index >= len(matrix):
        print(f"[DEBUG] Route node: current_step >= matrix length, finishing")
        return {"phase": "finish", "reason_code": None, "reason": None}
    views = _views(state)
    _set_status(views, index, "running", summary=None)
    step = matrix[index]
    print(f"[DEBUG] Route node: step interface={step.interface}")
    if step.interface == "GUI":
        phase = "execute_gui"
    elif step.interface == "CODING":
        phase = "execute_coding"
    else:  # CLI
        phase = "execute_cli"
    print(f"[DEBUG] Route node: routing to phase={phase}")
    return {"phase": phase, "step_views": _store_views(views)}


def _execute(state: AgentState, deps: GraphDeps, kind: str) -> dict:
    try:
        matrix = _matrix(state)
    except Exception as e:
        # Handle validation errors: mark current step as ERROR and skip subsequent steps
        print(f"[DEBUG] _matrix validation failed: {e}")
        import traceback
        traceback.print_exc()
        
        index = state.get("current_step", 0)
        views = _views(state)
        
        # Mark current step as error
        if index < len(views):
            _set_status(views, index, "error", assertion_passed=None, summary=str(e))
        
        # Skip all subsequent steps
        for later in range(index + 1, len(views)):
            _set_status(views, later, "skipped", assertion_passed=None)
        
        return {
            "phase": "finish",
            "reason_code": "engine_error",
            "reason": str(e),
            "step_views": _store_views(views),
        }
    
    index = state.get("current_step", 0)
    step = matrix[index]
    logger.info(f"=== Executing step {step.step} ({step.interface}) ===")
    logger.info(f"Step action: {str(step.action)[:100] if step.action else 'None'}...")
    logger.info(f"Step verifications: {len(step.verifications) if step.verifications else 0}")
    from pathlib import Path

    evidence_dir = Path(deps.evidence_dir_for(state["run_id"]))
    
    # Track retry attempts for this step
    attempt_counts = state.get("attempt_counts", {})
    step_key = f"{index}_{kind}"
    current_attempts = attempt_counts.get(step_key, 0)
    max_retries = state.get("max_retries", 3)
    
    try:
        if kind == "gui":
            result = deps.gui.execute_visual_action(step, evidence_dir)
        elif kind == "cli":
            result = deps.cli.execute_runtime_action(step, evidence_dir)
        else:  # coding
            # Check if CODING step contains execute_code - if so, delegate to CLI
            if step.coding_operations and any(op.action == "execute_code" for op in step.coding_operations):
                # Execute as CLI instead
                result = deps.cli.execute_runtime_action(step, evidence_dir)
            else:
                result = deps.coding.execute_coding_action(step, evidence_dir)
    except HarnessError as exc:
        return _handle_failure(state, deps, index, kind, step, exc.code, str(exc), current_attempts, max_retries)
    except PiAgentError as exc:
        return _handle_failure(state, deps, index, kind, step, exc.code, str(exc), current_attempts, max_retries)
    except Exception as exc:  # noqa: BLE001
        import traceback
        error_msg = f"{type(exc).__name__}: {exc!s}\n{traceback.format_exc()}"
        return _handle_failure(state, deps, index, kind, step, "engine_error", error_msg, current_attempts, max_retries)
    
    # Step succeeded, reset attempt count
    attempt_counts[step_key] = 0
    
    # Preserve coding_instructions in state
    coding_instructions = state.get("coding_instructions", [])
    
    return {
        "phase": "validate",
        "last_result": result.model_dump(),
        "attempt_counts": attempt_counts,
        "coding_instructions": coding_instructions,
    }


def _handle_failure(state: AgentState, deps: GraphDeps, index: int, kind: str, step: dict, code: str, message: str, current_attempts: int, max_retries: int) -> dict:
    """Handle step failure with retry logic."""
    step_key = f"{index}_{kind}"
    attempt_counts = state.get("attempt_counts", {})
    
    if current_attempts < max_retries:
        # Increment attempt count
        attempt_counts[step_key] = current_attempts + 1
        
        # Ask LLM to fix the step
        try:
            fixed_step = deps.planner.fix_step(step, message, current_attempts)
            
            # Check if LLM determined this should not be retried (hard failure)
            if fixed_step.get("_should_not_retry"):
                judgment = fixed_step.get("_retry_judgment", "Hard failure detected by LLM")
                return _error_update(state, index, code, f"Hard failure: {judgment}")
            
            # Update the matrix with the fixed step
            matrix = _matrix(state)
            matrix[index] = fixed_step
            
            # Set status to retrying
            views = _views(state)
            _set_status(views, index, "retrying", assertion_passed=None, summary=f"Retry {current_attempts + 1}/{max_retries}: {message}")
            
            return {
                "phase": kind,  # Retry the same phase
                "test_matrix": matrix,
                "attempt_counts": attempt_counts,
                "step_views": _store_views(views),
            }
        except Exception as exc:  # noqa: BLE001
            # If fixing fails, proceed to error
            import traceback
            error_msg = f"Failed to fix step: {exc}\n{traceback.format_exc()}"
            return _error_update(state, index, code, f"{message}\n\n{error_msg}")
    else:
        # Max retries reached, fail the step
        return _error_update(state, index, code, f"Failed after {max_retries} retries: {message}")


def execute_gui_node(state: AgentState, deps: GraphDeps) -> dict:
    return _execute(state, deps, "gui")


def execute_cli_node(state: AgentState, deps: GraphDeps) -> dict:
    return _execute(state, deps, "cli")


def execute_coding_node(state: AgentState, deps: GraphDeps) -> dict:
    # Capture coding instructions before execution for LLM context
    matrix = _matrix(state)
    index = state.get("current_step", 0)
    step = matrix[index]
    
    coding_instructions = state.get("coding_instructions", [])
    if step.coding_operations:
        # Store the LLM's instructions to the coding agent
        coding_instructions.append({
            "step": step.step,
            "action": step.action,
            "coding_operations": [op.model_dump() for op in step.coding_operations],
        })
    
    return _execute(state, deps, "coding")


def _error_update(state: AgentState, index: int, code: str, message: str) -> dict:
    views = _views(state)
    _set_status(views, index, "error", assertion_passed=None, summary=message)
    for later in range(index + 1, len(views)):
        _set_status(views, later, "skipped", assertion_passed=None)
    return {
        "phase": "finish",
        "reason_code": code,
        "reason": message,
        "step_views": _store_views(views),
    }


def _structured_check(question: str) -> bool:
    text = question.strip()
    return text.startswith(("contains:", "json:", "status:"))


def _command_text(evidence: dict) -> str:
    stdout = str(evidence.get("stdout") or "")
    stderr = str(evidence.get("stderr") or "")
    exit_code = evidence.get("exit_code")
    text = f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
    if exit_code is not None:
        text = f"{text}\n\nexit_code: {exit_code}"
    return text


def _judged_or_unreadable(deps: GraphDeps, question: str, page_source: str) -> Judgment:
    try:
        return _page_judgment(deps, question, page_source)
    except ValueError as exc:
        return Judgment(passed=False, judgment=str(exc))


def _page_judgment(deps: GraphDeps, question: str, page_source: str) -> Judgment:
    if deps.judge is None:
        deps.judge = build_judge()
    return deps.judge.judge(question, page_source)


def validate_node(state: AgentState, deps: GraphDeps) -> dict:
    matrix = _matrix(state)
    index = state.get("current_step", 0)
    step = matrix[index]
    result = ActionResult.model_validate(state.get("last_result") or {})
    evidence = dict(result.evidence)
    page_source = str(evidence.pop("page_source", "") or "")

    # For CLI steps with verifications, check execution result first before calling judge
    # If execution failed (non-zero exit code), trigger retry logic
    if step.interface == "CLI" and (step.assertion or step.verifications) and not result.ok:
        step_key = f"{index}_cli"
        attempt_counts = state.get("attempt_counts", {})
        current_attempts = attempt_counts.get(step_key, 0)
        max_retries = state.get("max_retries", 3)
        
        if current_attempts < max_retries:
            # Increment attempt count
            attempt_counts[step_key] = current_attempts + 1
            
            # Ask LLM to fix the step
            try:
                fixed_step = deps.planner.fix_step(step.model_dump(), result.summary, current_attempts)
                
                # Update the matrix with the fixed step
                matrix[index] = fixed_step
                
                # Set status to retrying
                views = _views(state)
                _set_status(views, index, "retrying", assertion_passed=None, summary=f"Retry {current_attempts + 1}/{max_retries}: {result.summary}")
                
                return {
                    "phase": "execute_cli",  # Retry the CLI execution
                    "test_matrix": matrix,
                    "attempt_counts": attempt_counts,
                    "step_views": _store_views(views),
                }
            except Exception as exc:  # noqa: BLE001
                # If fixing fails, proceed to normal verification (which will fail)
                import traceback
                logger.warning(f"Failed to fix step: {exc}")

    # For GUI steps with verifications, check execution result first before calling judge
    # If execution failed, trigger retry logic
    if step.interface == "GUI" and (step.assertion or step.verifications) and not result.ok:
        step_key = f"{index}_gui"
        attempt_counts = state.get("attempt_counts", {})
        current_attempts = attempt_counts.get(step_key, 0)
        max_retries = state.get("max_retries", 3)
        
        if current_attempts < max_retries:
            # Increment attempt count
            attempt_counts[step_key] = current_attempts + 1
            
            # Ask LLM to fix the step
            try:
                fixed_step = deps.planner.fix_step(step.model_dump(), result.summary, current_attempts)
                
                # Update the matrix with the fixed step
                matrix[index] = fixed_step
                
                # Set status to retrying
                views = _views(state)
                _set_status(views, index, "retrying", assertion_passed=None, summary=f"Retry {current_attempts + 1}/{max_retries}: {result.summary}")
                
                return {
                    "phase": "execute_gui",  # Retry the GUI execution
                    "test_matrix": matrix,
                    "attempt_counts": attempt_counts,
                    "step_views": _store_views(views),
                }
            except Exception as exc:  # noqa: BLE001
                # If fixing fails, proceed to normal verification (which will fail)
                import traceback
                logger.warning(f"Failed to fix step: {exc}")

    # Handle CODING steps with assertions
    # For CODING, if the operation succeeded, the assertion passes
    # The assertion is more of a description than something to verify against evidence
    if step.interface == "CODING" and (step.assertion or step.verifications):
        passed = result.ok
        judgment_text = result.summary if not passed else None
        results: list[VerificationResult] = []
        if step.assertion:
            results.append(VerificationResult(question=step.assertion, passed=passed, judgment=judgment_text))
        for verification in step.verifications or []:
            results.append(VerificationResult(question=verification, passed=passed, judgment=judgment_text))
        
        # If CODING operation failed, trigger retry logic
        if not passed:
            step_key = f"{index}_coding"
            attempt_counts = state.get("attempt_counts", {})
            current_attempts = attempt_counts.get(step_key, 0)
            max_retries = state.get("max_retries", 3)
            
            if current_attempts < max_retries:
                # Increment attempt count
                attempt_counts[step_key] = current_attempts + 1
                
                # Ask LLM to fix the step
                try:
                    fixed_step = deps.planner.fix_step(step.model_dump(), judgment_text, current_attempts)
                    
                    # Update the matrix with the fixed step
                    matrix[index] = fixed_step
                    
                    # Set status to retrying
                    views = _views(state)
                    _set_status(views, index, "retrying", assertion_passed=None, summary=f"Retry {current_attempts + 1}/{max_retries}: {judgment_text}")
                    
                    return {
                        "phase": "execute_coding",  # Retry the CODING execution
                        "test_matrix": matrix,
                        "attempt_counts": attempt_counts,
                        "step_views": _store_views(views),
                    }
                except Exception as exc:  # noqa: BLE001
                    # If fixing fails, proceed to failure
                    import traceback
                    error_msg = f"Failed to fix step: {exc}\n{traceback.format_exc()}"
                    judgment_text = f"{judgment_text}\n\n{error_msg}"
        
        # If CODING operation succeeded, update planner context with coding instructions
        if passed:
            coding_instructions = state.get("coding_instructions", [])
            if coding_instructions:
                deps.planner.update_coding_context(coding_instructions)
        
        views = _views(state)
        history = list(state.get("execution_history", []))
        history.append(
            {
                "task": step.model_dump(),
                "result": result.model_dump(),
                "assertion_passed": passed,
                "judgment": judgment_text,
            }
        )
        
        # Reset attempt count on success
        attempt_counts = state.get("attempt_counts", {})
        if passed:
            step_key = f"{index}_{step.interface.lower()}"
            attempt_counts[step_key] = 0
        
        if passed:
            _set_status(
                views,
                index,
                "passed",
                assertion_passed=True,
                judgment=judgment_text,
                verification_results=results,
                summary=result.summary,
                evidence=evidence,
            )
            return {
                "phase": "route",
                "current_step": index + 1,
                "step_views": _store_views(views),
                "execution_history": history,
                "attempt_counts": attempt_counts,
                "last_result": result.model_copy(update={"assertion_passed": True}).model_dump(),
            }
        else:
            _set_status(
                views,
                index,
                "failed",
                assertion_passed=False,
                judgment=judgment_text,
                verification_results=results,
                summary=result.summary,
                evidence=evidence,
            )
            for later in range(index + 1, len(views)):
                _set_status(views, later, "skipped", assertion_passed=None)
            reason = result.summary or f"Step {step.step} execution failed"
            return {
                "phase": "finish",
                "reason_code": "assertion_failed",
                "reason": reason,
                "step_views": _store_views(views),
                "execution_history": history,
            }

    # Handle steps without assertions (non-testing steps)
    if not step.assertion and not step.verifications:
        # For non-testing steps, success is determined by execution success
        # For CLI steps, pass if and only if exit code ($?) is 0
        if step.interface == "CLI":
            passed = result.ok  # result.ok is True when exit code is 0
            judgment_text = result.summary if not passed else "Command executed successfully"
            
            # If CLI command failed, trigger retry logic
            if not passed:
                step_key = f"{index}_cli"
                attempt_counts = state.get("attempt_counts", {})
                current_attempts = attempt_counts.get(step_key, 0)
                max_retries = state.get("max_retries", 3)
                
                if current_attempts < max_retries:
                    # Increment attempt count
                    attempt_counts[step_key] = current_attempts + 1
                    
                    # Ask LLM to fix the step
                    try:
                        fixed_step = deps.planner.fix_step(step.model_dump(), judgment_text, current_attempts)
                        
                        # Update the matrix with the fixed step
                        matrix[index] = fixed_step
                        
                        # Set status to retrying
                        views = _views(state)
                        _set_status(views, index, "retrying", assertion_passed=None, summary=f"Retry {current_attempts + 1}/{max_retries}: {judgment_text}")
                        
                        return {
                            "phase": "execute_cli",  # Retry the CLI execution
                            "test_matrix": matrix,
                            "attempt_counts": attempt_counts,
                            "step_views": _store_views(views),
                        }
                    except Exception as exc:  # noqa: BLE001
                        # If fixing fails, proceed to failure
                        import traceback
                        error_msg = f"Failed to fix step: {exc}\n{traceback.format_exc()}"
                        judgment_text = f"{judgment_text}\n\n{error_msg}"
            
            results: list[VerificationResult] = []
            results.append(VerificationResult(
                question="Command execution",
                passed=passed,
                judgment=judgment_text
            ))
        else:
            # For non-CLI steps, use result.ok
            passed = result.ok
            judgment_text = result.summary if not passed else None
            results: list[VerificationResult] = []
        
        views = _views(state)
        history = list(state.get("execution_history", []))
        history.append(
            {
                "task": step.model_dump(),
                "result": result.model_dump(),
                "assertion_passed": passed,
                "judgment": judgment_text,
            }
        )
        
        # Reset attempt count on success
        attempt_counts = state.get("attempt_counts", {})
        if passed:
            step_key = f"{index}_{step.interface.lower()}"
            attempt_counts[step_key] = 0
        
        if passed:
            _set_status(
                views,
                index,
                "passed",
                assertion_passed=True,
                judgment=judgment_text,
                verification_results=results,
                summary=result.summary,
                evidence=evidence,
            )
            return {
                "phase": "route",
                "current_step": index + 1,
                "step_views": _store_views(views),
                "execution_history": history,
                "attempt_counts": attempt_counts,
                "last_result": result.model_copy(update={"assertion_passed": True}).model_dump(),
            }
        else:
            _set_status(
                views,
                index,
                "failed",
                assertion_passed=False,
                judgment=judgment_text,
                verification_results=results,
                summary=result.summary,
                evidence=evidence,
            )
            for later in range(index + 1, len(views)):
                _set_status(views, later, "skipped", assertion_passed=None)
            reason = result.summary or f"Step {step.step} execution failed"
            return {
                "phase": "finish",
                "reason_code": "assertion_failed",
                "reason": reason,
                "step_views": _store_views(views),
                "execution_history": history,
            }

    # Handle steps with assertions (testing steps)
    questions = step.verifications or [step.assertion]
    logger.info(f"=== Processing {len(questions)} verification(s) for step {step.step} ===")
    results = []
    judgments: list[str] = []
    passed = True
    for question in questions:
        logger.info(f"--- Verification: {question[:100]}...")
        output = "\n".join(
            part
            for part in (str(evidence.get("stdout") or ""), str(evidence.get("stderr") or ""))
            if part.strip()
        )
        
        # Check if this is a file-based verification (mentions file, log, etc.)
        is_file_check = any(keyword in question.lower() for keyword in ["file", "log", "exists", "contains"])
        # Check if this is a port-based verification (mentions port, listening, lsof)
        is_port_check = any(keyword in question.lower() for keyword in ["port", "listening", "lsof", "socket"])
        
        logger.info(f"is_file_check: {is_file_check}, is_port_check: {is_port_check}")
        
        if step.interface == "CLI":
            # For CLI steps, don't use PageJudge - use direct evaluation
            if _structured_check(question):
                ok = evaluate_assertion(question, evidence)
                results.append(VerificationResult(question=question, passed=ok))
                passed = passed and ok
            elif is_port_check:
                # Port-based checks: actually run lsof to check the port
                # Extract port number from the question
                import re
                import subprocess
                port_match = re.search(r'port\s*(\d+)', question.lower())
                if port_match:
                    port = port_match.group(1)
                    # Run lsof to check the port
                    try:
                        result = subprocess.run(
                            ["lsof", f"-i:{port}", "-sTCP:LISTEN"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                            check=False
                        )
                        port_output = result.stdout or ""
                        if f":{port}" in port_output and "LISTEN" in port_output:
                            judgment = f"Port {port} is listening"
                            port_passed = True
                        else:
                            judgment = f"Port {port} is not listening"
                            port_passed = False
                    except Exception as exc:  # noqa: BLE001
                        judgment = f"Failed to check port {port}: {exc}"
                        port_passed = False
                else:
                    judgment = "Could not extract port number from verification"
                    port_passed = False
                results.append(VerificationResult(question=question, passed=port_passed, judgment=judgment))
                passed = passed and port_passed
            elif is_file_check:
                # File-based checks should use LLM judge to properly detect errors
                # Try to read the file content if it's mentioned in the verification
                file_content = output
                import re
                # Try to extract file path from the verification
                file_match = re.search(r'(\S+\.(?:log|txt|json|py|yaml|yml|conf|cfg))', question, re.IGNORECASE)
                if file_match:
                    file_path = file_match.group(1)
                    try:
                        # Try to read the file from the work directory
                        if deps.work_dir:
                            full_path = deps.work_dir / file_path
                            if full_path.exists():
                                file_content = full_path.read_text()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(f"Failed to read file {file_path}: {exc}")
                judged = _judged_or_unreadable(deps, question, file_content)
                results.append(
                    VerificationResult(question=question, passed=judged.passed, judgment=judged.judgment)
                )
                judgments.append(judged.judgment)
                passed = passed and judged.passed
            else:
                # For non-structured CLI assertions, check if command succeeded
                # If exit code is non-zero, it's an error
                stderr = str(evidence.get("stderr") or "")
                command_passed = result.ok
                if not result.ok:
                    judgment = "Command failed with exit code"
                # Use LLM to judge if stderr represents an actual error when exit code is 0
                elif stderr.strip():
                    if deps.chat_model is None:
                        from aqe.chat import get_chat_model
                        deps.chat_model = get_chat_model()
                    is_error, explanation = judge_stderr(deps.chat_model, stderr, 0)
                    if is_error:
                        command_passed = False
                        judgment = f"Command failed with error: {explanation}"
                    else:
                        judgment = f"Command succeeded with warnings: {explanation}"
                else:
                    judgment = "Command succeeded"
                results.append(VerificationResult(
                    question=question,
                    passed=command_passed,
                    judgment=judgment
                ))
                passed = passed and command_passed
        elif page_source.strip():
            judged = _judged_or_unreadable(deps, question, page_source)
            results.append(
                VerificationResult(question=question, passed=judged.passed, judgment=judged.judgment)
            )
            judgments.append(judged.judgment)
            passed = passed and judged.passed
        elif output.strip() and not _structured_check(question):
            judged = _judged_or_unreadable(deps, question, output)
            results.append(
                VerificationResult(question=question, passed=judged.passed, judgment=judged.judgment)
            )
            judgments.append(judged.judgment)
            passed = passed and judged.passed
        else:
            ok = evaluate_assertion(question, evidence)
            results.append(VerificationResult(question=question, passed=ok))
            passed = passed and ok
    judgment_text = "\n".join(item for item in judgments if item) or None
    views = _views(state)
    history = list(state.get("execution_history", []))
    history.append(
        {
            "task": step.model_dump(),
            "result": result.model_dump(),
            "assertion_passed": passed,
            "judgment": judgment_text,
        }
    )
    if passed:
        _set_status(
            views,
            index,
            "passed",
            assertion_passed=True,
            judgment=judgment_text,
            verification_results=results,
            summary=result.summary,
            evidence=evidence,
        )
        return {
            "phase": "route",
            "current_step": index + 1,
            "step_views": _store_views(views),
            "execution_history": history,
            "last_result": result.model_copy(update={"assertion_passed": True}).model_dump(),
        }
    attempts = dict(state.get("attempt_counts", {}))
    key = str(index)
    attempts[key] = attempts.get(key, 0) + 1
    if attempts[key] <= deps.config.max_retries:
        _set_status(
            views,
            index,
            "retrying",
            assertion_passed=False,
            judgment=judgment_text,
            verification_results=results,
            summary=result.summary,
            evidence=evidence,
        )
        return {
            "phase": "reflect",
            "attempt_counts": attempts,
            "step_views": _store_views(views),
            "execution_history": history,
        }
    _set_status(
        views,
        index,
        "failed",
        assertion_passed=False,
        judgment=judgment_text,
        verification_results=results,
        summary=result.summary,
        evidence=evidence,
    )
    for later in range(index + 1, len(views)):
        _set_status(views, later, "skipped", assertion_passed=None)
    failed = [item for item in results if item.passed is False]
    reason = "\n".join(item.judgment or f"Verification failed: {item.question}" for item in failed)
    reason = reason or judgment_text or result.summary or f"Step {step.step} did not satisfy {step.assertion}"
    return {
        "phase": "finish",
        "reason_code": "assertion_failed",
        "reason": reason,
        "attempt_counts": attempts,
        "step_views": _store_views(views),
        "execution_history": history,
    }


def reflect_node(state: AgentState, deps: GraphDeps) -> dict:
    del deps
    return {"phase": "route"}


def finish_node(state: AgentState, deps: GraphDeps) -> dict:
    views = _views(state)
    reason_code = state.get("reason_code")
    if reason_code == "canceled":
        index = state.get("current_step", 0)
        for cursor, view in enumerate(views):
            if cursor >= index and view.status in {"pending", "running", "retrying"}:
                _set_status(views, cursor, "skipped", assertion_passed=None)
        verdict = "canceled"
    elif reason_code == "missing_capability":
        for cursor, view in enumerate(views):
            if view.status == "pending":
                _set_status(views, cursor, "skipped", assertion_passed=None)
        verdict = "rejected"
    elif reason_code == "not_a_test_plan":
        verdict = "rejected"
    elif reason_code == "assertion_failed":
        verdict = "fail"
    elif reason_code in {
        "sandbox_start_failed",
        "browser_launch_failed",
        "desktop_input_failed",
        "driver_timeout",
        "engine_error",
        "coding_agent_failed",
        "coding_agent_not_available",
    }:
        verdict = "error"
    elif any(view.status == "error" for view in views):
        verdict = "error"
        reason_code = reason_code or "engine_error"
    elif views and all(view.assertion_passed is True for view in views):
        verdict = "pass"
        reason_code = None
    elif views and all(view.status == "passed" for view in views):
        # All operations succeeded even without assertions
        verdict = "pass"
        reason_code = None
    elif not views and reason_code is None:
        verdict = "rejected"
        reason_code = "not_a_test_plan"
    else:
        verdict = "fail"
        reason_code = reason_code or "assertion_failed"
    report = TestReport(
        id=state["run_id"],
        verdict=verdict,
        specification=state.get("specification") or "",
        reason_code=reason_code,
        reason=state.get("reason"),
        missing=list(state.get("missing") or []),
        steps=views,
    )

    return {"phase": "done", "report": report.model_dump(), "step_views": _store_views(views)}


def _route(state: AgentState) -> str:
    phase = state.get("phase", "finish")
    if phase == "done":
        return END
    return phase


def build_graph(deps: GraphDeps, publish: Callable[[AgentState], None] | None = None):
    def bind(fn):
        def node(state: AgentState) -> dict:
            try:
                update = fn(state, deps)
                if publish is not None:
                    publish({**state, **update})
                return update
            except Exception as e:
                # Handle errors: mark current step as ERROR and skip subsequent steps
                print(f"[DEBUG] Node {fn.__name__} failed with error: {e}")
                import traceback
                traceback.print_exc()
                
                # Get current step index
                matrix = _matrix(state)
                index = state.get("current_step", 0)
                views = _views(state)
                
                # Mark current step as error
                if index < len(views):
                    _set_status(views, index, "error", assertion_passed=None, summary=str(e))
                
                # Skip all subsequent steps
                for later in range(index + 1, len(views)):
                    _set_status(views, later, "skipped", assertion_passed=None)
                
                error_update = {
                    "phase": "finish",
                    "reason_code": "engine_error",
                    "reason": str(e),
                    "step_views": _store_views(views),
                }
                
                if publish is not None:
                    publish({**state, **error_update})
                return error_update

        return node

    graph = StateGraph(AgentState)
    graph.add_node("plan", bind(plan_node))
    graph.add_node("preflight", bind(preflight_node))
    graph.add_node("route", bind(route_node))
    graph.add_node("execute_gui", bind(execute_gui_node))
    graph.add_node("execute_cli", bind(execute_cli_node))
    graph.add_node("execute_coding", bind(execute_coding_node))
    graph.add_node("validate", bind(validate_node))
    graph.add_node("reflect", bind(reflect_node))
    graph.add_node("finish", bind(finish_node))
    graph.set_entry_point("plan")
    for name in ("plan", "preflight", "route", "execute_gui", "execute_cli", "execute_coding", "validate", "reflect", "finish"):
        graph.add_conditional_edges(name, _route)
    return graph.compile()


def initial_state(run_id: str, specification: str) -> AgentState:
    return {
        "run_id": run_id,
        "specification": specification,
        "current_step": 0,
        "test_matrix": [],
        "execution_history": [],
        "attempt_counts": {},
        "phase": "plan",
        "step_views": [],
        "last_result": None,
        "max_retries": 3,
        "report": None,
        "reason_code": None,
        "reason": None,
        "missing": [],
        "coding_instructions": [],
    }


def run_graph(deps: GraphDeps, state: AgentState, publish: Callable[[AgentState], None]) -> AgentState:
    """Run the graph and publish the merged state after every node."""
    compiled = build_graph(deps, publish)
    return compiled.invoke(dict(state))


def default_probe_for(config: EngineConfig) -> Probe:
    return lambda: probe_host(config)
