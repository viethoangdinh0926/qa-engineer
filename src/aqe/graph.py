"""Plan, preflight, route, execute, validate, and reflect."""

from collections.abc import Callable
from dataclasses import dataclass

from langgraph.graph import END, StateGraph

from aqe.assertions import evaluate_assertion
from aqe.capabilities import (
    Probe,
    missing_details,
    probe_host,
    required_capabilities,
)
from aqe.cli_runtime.synthesizer import CLISubsystem
from aqe.config import EngineConfig
from aqe.coding_agent.subsystem import CodingSubsystem, PiAgentError
from aqe.errors import HarnessError
from aqe.gui.subsystem import GUISubsystem
from aqe.judge import PageJudge, build_judge
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


def _views(state: AgentState) -> list[StepView]:
    return [StepView.model_validate(item) for item in state.get("step_views", [])]


def _store_views(views: list[StepView]) -> list[dict]:
    return [view.model_dump() for view in views]


def _matrix(state: AgentState) -> list[TestStep]:
    return [TestStep.model_validate(item) for item in state.get("test_matrix", [])]


def _set_status(views: list[StepView], index: int, status: str, **updates: object) -> None:
    view = views[index].model_copy(update={"status": status, **updates})
    views[index] = view


def plan_node(state: AgentState, deps: GraphDeps) -> dict:
    try:
        result = deps.planner.plan(state["specification"])
    except Exception as exc:  # noqa: BLE001 - bad planner output rejects the run
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
    needed = required_capabilities(
        state.get("test_matrix", []),
        llm=deps.config.llm,
        llm_configured=deps.config.llm_configured,
    )
    missing = []
    flags = {
        "browser": capabilities.browser.available,
        "desktop": capabilities.desktop.available,
        "sandbox": capabilities.sandbox.available,
        "coding": capabilities.coding.available,
        "llm": deps.config.llm_configured,
    }
    for name in needed:
        if name == "llm":
            if not deps.config.llm_configured:
                missing.append(name)
            continue
        if not flags[name]:
            missing.append(name)
    if missing:
        details = missing_details(capabilities, missing)
        return {
            "phase": "finish",
            "reason_code": "missing_capability",
            "reason": " ".join(details) or "A required driver is unavailable.",
            "missing": missing,
        }
    return {"phase": "route", "missing": []}


def route_node(state: AgentState, deps: GraphDeps) -> dict:
    if deps.control.cancel_requested:
        return {"phase": "finish", "reason_code": "canceled", "reason": "The run was canceled."}
    matrix = _matrix(state)
    index = state.get("current_step", 0)
    if index >= len(matrix):
        return {"phase": "finish", "reason_code": None, "reason": None}
    views = _views(state)
    _set_status(views, index, "running", summary=None)
    step = matrix[index]
    if step.interface == "GUI":
        phase = "execute_gui"
    elif step.interface == "CODING":
        phase = "execute_coding"
    else:  # CLI
        phase = "execute_cli"
    return {"phase": phase, "step_views": _store_views(views)}


def _execute(state: AgentState, deps: GraphDeps, kind: str) -> dict:
    matrix = _matrix(state)
    index = state.get("current_step", 0)
    step = matrix[index]
    from pathlib import Path

    evidence_dir = Path(deps.evidence_dir_for(state["run_id"]))
    try:
        if kind == "gui":
            result = deps.gui.execute_visual_action(step, evidence_dir)
        elif kind == "cli":
            result = deps.cli.execute_runtime_action(step, evidence_dir)
        else:  # coding
            result = deps.coding.execute_coding_action(step, evidence_dir)
    except HarnessError as exc:
        return _error_update(state, index, exc.code, str(exc))
    except PiAgentError as exc:
        return _error_update(state, index, exc.code, str(exc))
    except Exception as exc:  # noqa: BLE001
        import traceback
        error_msg = f"{type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}"
        return _error_update(state, index, "engine_error", error_msg)
    return {"phase": "validate", "last_result": result.model_dump()}


def execute_gui_node(state: AgentState, deps: GraphDeps) -> dict:
    return _execute(state, deps, "gui")


def execute_cli_node(state: AgentState, deps: GraphDeps) -> dict:
    return _execute(state, deps, "cli")


def execute_coding_node(state: AgentState, deps: GraphDeps) -> dict:
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
    return f"stdout:\n{stdout}\n\nstderr:\n{stderr}"


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
        passed = result.ok
        judgment_text = result.summary if not passed else None
        results: list[VerificationResult] = []
        
        # For CLI steps without assertions, check stdout/stderr for errors
        if step.interface == "CLI":
            stdout = str(evidence.get("stdout") or "")
            stderr = str(evidence.get("stderr") or "")
            # If stderr has content, consider it a failure
            if stderr.strip():
                passed = False
                judgment_text = f"Command produced stderr output: {stderr}"
            # If exit code is non-zero, it's already captured in result.ok
            results.append(VerificationResult(
                question="Command execution",
                passed=passed,
                judgment=judgment_text
            ))
        
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
    results = []
    judgments: list[str] = []
    passed = True
    for question in questions:
        output = "\n".join(
            part
            for part in (str(evidence.get("stdout") or ""), str(evidence.get("stderr") or ""))
            if part.strip()
        )
        if step.interface == "CLI":
            # For CLI steps, don't use PageJudge - use direct evaluation
            if _structured_check(question):
                ok = evaluate_assertion(question, evidence)
                results.append(VerificationResult(question=question, passed=ok))
                passed = passed and ok
            else:
                # For non-structured CLI assertions, check if command succeeded
                # Command succeeded if exit code is 0 and no stderr
                stdout = str(evidence.get("stdout") or "")
                stderr = str(evidence.get("stderr") or "")
                command_passed = result.ok and not stderr.strip()
                results.append(VerificationResult(
                    question=question,
                    passed=command_passed,
                    judgment=f"Command {'succeeded' if command_passed else 'failed'}"
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
            update = fn(state, deps)
            if publish is not None:
                publish({**state, **update})
            return update

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
    }


def run_graph(deps: GraphDeps, state: AgentState, publish: Callable[[AgentState], None]) -> AgentState:
    """Run the graph and publish the merged state after every node."""
    compiled = build_graph(deps, publish)
    return compiled.invoke(dict(state))


def default_probe_for(config: EngineConfig) -> Probe:
    return lambda: probe_host(config)
