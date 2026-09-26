"""Plan, preflight, route, execute, validate, and reflect."""

import json
import logging
import subprocess
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
from aqe.judge import PageJudge, build_judge
from aqe.llm import Planner, _coding_operations_from
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
                    
                    if cli_commands and not str(item.get("action") or "").strip():
                        item["action"] = " && ".join(command for command in cli_commands if command)
                    if cli_commands or gui_actions:
                        item["operations"] = gui_actions
                
                item["coding_operations"] = [
                    operation.model_dump()
                    for operation in _coding_operations_from(item.get("coding_operations") or [])
                ]
            
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
    if state.get("test_matrix"):
        steps = _matrix(state)
        views = [StepView.from_step(step) for step in steps]
        return {
            "phase": "preflight",
            "test_matrix": [step.model_dump() for step in steps],
            "step_views": _store_views(views),
            "current_step": 0,
            "reason_code": None,
            "reason": None,
        }
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
        return _error_update(state, index, exc.code, str(exc))
    except PiAgentError as exc:
        return _error_update(state, index, exc.code, str(exc))
    except Exception as exc:  # noqa: BLE001
        import traceback
        error_msg = f"{type(exc).__name__}: {exc!s}\n{traceback.format_exc()}"
        return _error_update(state, index, "engine_error", error_msg)

    coding_instructions = state.get("coding_instructions", [])
    return {
        "phase": "validate",
        "last_result": result.model_dump(),
        "attempt_counts": state.get("attempt_counts", {}),
        "coding_instructions": coding_instructions,
    }


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



def _structured_check(question: str) -> bool:
    text = question.strip()
    return text.startswith(("contains:", "json:", "status:"))


def _cli_exit_code(result: ActionResult, evidence: dict) -> int:
    """$? for a CLI step. A missing code follows whether the runner reported success."""
    raw = evidence.get("exit_code")
    if raw is None:
        return 0 if result.ok else 1
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0 if result.ok else 1


def _cli_packet(question: str, exit_code: int, stdout: str, stderr: str) -> str:
    return (
        f"Verification statement:\n{question.strip()}\n\n"
        f"$?: {exit_code}\n\n"
        f"stdout:\n{stdout}\n\n"
        f"stderr:\n{stderr}\n"
    )


def _run_verify_script(work_dir: Path, script: str) -> tuple[int, str, str]:
    """Run the verification command the model requested, in the phase directory."""
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "verify.sh"
    path.write_text(script if script.endswith("\n") else f"{script}\n", encoding="utf-8")
    try:
        result = subprocess.run(
            ["bash", str(path)],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, "", "Verification command timed out"
    except OSError as exc:
        return 1, "", str(exc)
    return result.returncode, result.stdout, result.stderr


def _cli_verification(deps: GraphDeps, question: str, result: ActionResult, evidence: dict) -> Judgment:
    """Ask the model to decide a CLI check, or to name a command the agent should run."""
    exit_code = _cli_exit_code(result, evidence)
    stdout = str(evidence.get("stdout") or "")
    stderr = str(evidence.get("stderr") or "")
    judge = deps.judge
    if judge is None:
        judge = build_judge()
        deps.judge = judge
    if hasattr(judge, "judge_cli"):
        check = judge.judge_cli(question, exit_code, stdout, stderr)
        if check.script.strip():
            work = deps.work_dir
            if work is None:
                return Judgment(
                    passed=False,
                    judgment=check.judgment or "The verification command has no work directory.",
                )
            code, out, err = _run_verify_script(Path(work), check.script)
            check = judge.judge_cli(question, code, out, err, follow_up=True)
        if check.passed is None:
            return Judgment(passed=False, judgment=check.judgment or "The model did not decide the verification.")
        return Judgment(passed=check.passed, judgment=check.judgment)
    return _judged_or_unreadable(deps, question, _cli_packet(question, exit_code, stdout, stderr))


def _command_text(evidence: dict) -> str:
    stdout = str(evidence.get("stdout") or "")
    stderr = str(evidence.get("stderr") or "")
    exit_code = evidence.get("exit_code")
    text = f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
    if exit_code is not None:
        text = f"{text}\n\nexit_code: {exit_code}"
    script_result = str(evidence.get("script_result") or "").strip()
    if script_result:
        text = f"{text}\n\nscript_result:\n{script_result}\n"
    return text


def _coding_text(evidence: dict) -> str:
    """Show the files a coding phase wrote so each verification can be judged from them."""
    results = evidence.get("results")
    blocks: list[str] = []
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            lines: list[str] = []
            action = str(item.get("action") or "").strip()
            path = str(item.get("file_path") or "").strip()
            if action:
                lines.append(f"action: {action}")
            if path:
                lines.append(f"file: {path}")
            error = str(item.get("error") or "").strip()
            if item.get("success") is False or error:
                lines.append(f"error: {error or 'the coding operation failed'}")
            elif "content" in item:
                content = str(item.get("content") or "")
                lines.append(f"content length: {len(content)} characters")
                lines.append(f"content:\n{content}")
            if lines:
                blocks.append("\n".join(lines))
    if blocks:
        return "\n\n".join(blocks)
    if any(key in evidence for key in ("stdout", "stderr", "exit_code")):
        return _command_text(evidence)
    errors = evidence.get("errors")
    if isinstance(errors, list) and errors:
        return "errors:\n" + "\n".join(str(item) for item in errors)
    return str(evidence.get("summary") or "")


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

    # Handle steps without assertions (non-testing steps)
    if not step.assertion and not step.verifications:
        # For non-testing steps, success is determined by execution success
        # For CLI steps, pass if and only if exit code ($?) is 0
        if step.interface == "CLI":
            passed = result.ok  # result.ok is True when exit code is 0
            judgment_text = result.summary if not passed else "Command executed successfully"
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
        
        if step.interface == "CLI":
            judged = _cli_verification(deps, question, result, evidence)
            results.append(
                VerificationResult(question=question, passed=judged.passed, judgment=judged.judgment)
            )
            judgments.append(judged.judgment)
            passed = passed and judged.passed
        elif step.interface == "CODING":
            judged = _judged_or_unreadable(deps, question, _coding_text(evidence))
            results.append(
                VerificationResult(question=question, passed=judged.passed, judgment=judged.judgment)
            )
            judgments.append(judged.judgment)
            passed = passed and judged.passed
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
        attempt_counts = dict(state.get("attempt_counts") or {})
        attempt_counts[f"{index}_{step.interface.lower()}"] = 0
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
            "attempt_counts": attempt_counts,
            "step_views": _store_views(views),
            "execution_history": history,
            "last_result": result.model_copy(update={"assertion_passed": True}).model_dump(),
        }
    reason = judgment_text or result.summary or f"Step {step.step} did not satisfy {step.assertion}"
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
        "attempt_counts": dict(state.get("attempt_counts") or {}),
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

                index = state.get("current_step", 0)
                try:
                    views = _views(state)
                except Exception:
                    logger.exception("Could not read step views after %s failed", fn.__name__)
                    views = []
                
                # Mark current step as error
                if index < len(views):
                    _set_status(views, index, "error", assertion_passed=None, summary=str(e))
                
                # Skip all subsequent steps
                for later in range(index + 1, len(views)):
                    _set_status(views, later, "skipped", assertion_passed=None)
                
                stored_views = _store_views(views)
                if fn.__name__ == "finish_node":
                    report = TestReport(
                        id=state.get("run_id", ""),
                        verdict="error",
                        specification=state.get("specification") or "",
                        reason_code="engine_error",
                        reason=str(e),
                        missing=list(state.get("missing") or []),
                        steps=views,
                    )
                    error_update = {
                        "phase": "done",
                        "reason_code": "engine_error",
                        "reason": str(e),
                        "report": report.model_dump(),
                        "step_views": stored_views,
                    }
                else:
                    error_update = {
                        "phase": "finish",
                        "reason_code": "engine_error",
                        "reason": str(e),
                        "step_views": stored_views,
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
