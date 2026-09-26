"""In-memory runs. Submit returns immediately; the graph runs on a background thread."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aqe.capabilities import Probe, probe_host
from aqe.cli_runtime.synthesizer import CLISubsystem
from aqe.coding_agent.subsystem import CodingSubsystem, build_coding_agent
from aqe.config import EngineConfig
from aqe.errors import AgentBusyError, SpecValidationError
from aqe.graph import GraphDeps, RunControl, initial_state, run_graph
from aqe.gui.subsystem import GUISubsystem, build_gui
from aqe.judge import PageJudge
from aqe.llm import Planner, build_planner
from aqe.plan_storage import PlanStorageManager
from aqe.state import AgentState, PlanStorage, PlannerChatHistory, TERMINAL_STATUSES, status_for_verdict

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    id: str
    specification: str
    status: str = "submitted"
    ready: bool = False
    steps: list[dict[str, Any]] = field(default_factory=list)
    report: dict[str, Any] | None = None
    execution_history: list[dict[str, Any]] = field(default_factory=list)
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    control: RunControl = field(default_factory=RunControl)
    done: threading.Event = field(default_factory=threading.Event)
    error: str | None = None
    generation: int = 0


class RunService:
    def __init__(
        self,
        config: EngineConfig | None = None,
        *,
        planner: Planner | None = None,
        gui: GUISubsystem | None = None,
        coding: CodingSubsystem | None = None,
        judge: PageJudge | None = None,
        probe: Probe | None = None,
        pause: threading.Event | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.config.runs_dir.mkdir(parents=True, exist_ok=True)
        self.planner = planner
        self.gui = gui
        self.coding = coding
        self.judge = judge
        self.probe = probe or (lambda: probe_host(self.config))
        self.pause = pause
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()
        self._activity = threading.Lock()
        self._load_saved_runs()

    def submit(self, specification: str, *, run_id: str | None = None, wait_for_approval: bool = False) -> dict[str, Any]:
        self._validate(specification)
        if not self._activity.acquire(blocking=False):
            raise AgentBusyError()
        record = RunRecord(
            id=run_id or uuid.uuid4().hex,
            specification=specification,
        )
        try:
            with self._lock:
                self._runs[record.id] = record
            self._remember(record)

            try:
                planner = self._planner()
            except Exception as exc:  # noqa: BLE001 - a missing client is an engine error
                self._finish_error(record, str(exc))
                return self.snapshot(record)
            try:
                plan_result = planner.plan(specification)
            except Exception as exc:  # noqa: BLE001 - a planner crash is a rejected run
                self._finish_rejected(
                    record,
                    f"planner output did not match the test plan schema: {exc}",
                )
                return self.snapshot(record)
            if not plan_result.accepted:
                self._finish_rejected(
                    record,
                    plan_result.reason or "This request has no verifiable GUI or CLI actions.",
                    plan_result.reason_code or "not_a_test_plan",
                )
                return self.snapshot(record)

            plan_storage = PlanStorageManager(self.config.runs_dir)
            plan = PlanStorage(
                run_id=record.id,
                status="draft" if wait_for_approval else "approved",
                phases=plan_result.phases,
                steps=plan_result.steps,
            )
            plan_storage.save_plan(plan)
            if wait_for_approval:
                record.status = "submitted"
                self._remember(record)
                return self.snapshot(record)
        finally:
            self._activity.release()

        record.status = "running"
        record.ready = False
        self._remember(record)
        thread = threading.Thread(target=self._execute, args=(record.id,), daemon=True)
        thread.start()
        return self.snapshot(record)

    def _planner(self) -> Planner:
        if self.planner is None:
            self.planner = build_planner(self.config)
        return self.planner

    def _finish_error(self, record: RunRecord, reason: str) -> None:
        record.status = "failed"
        record.ready = True
        record.error = reason
        record.report = {
            "schema_version": "1",
            "id": record.id,
            "verdict": "error",
            "specification": record.specification,
            "reason_code": "engine_error",
            "reason": reason,
            "missing": [],
            "steps": record.steps,
        }
        self._write_report(record)
        self._remember(record)
        record.done.set()

    def _finish_rejected(self, record: RunRecord, reason: str, reason_code: str = "not_a_test_plan") -> None:
        record.status = "rejected"
        record.ready = True
        record.report = {
            "schema_version": "1",
            "id": record.id,
            "verdict": "rejected",
            "specification": record.specification,
            "reason_code": reason_code,
            "reason": reason,
            "missing": [],
            "steps": [],
        }
        self._write_report(record)
        self._remember(record)
        record.done.set()

    def get(self, run_id: str) -> dict[str, Any] | None:
        record = self._runs.get(run_id)
        if record is None:
            return None
        return self.snapshot(record)

    def list_runs(self) -> list[dict[str, Any]]:
        return [self.snapshot(record) for record in reversed(list(self._runs.values()))]

    def cancel(self, run_id: str) -> dict[str, Any] | None:
        record = self._runs.get(run_id)
        if record is None:
            return None
        record.control.cancel_requested = True
        return self.snapshot(record)

    def agent_busy(self) -> bool:
        if not self._activity.acquire(blocking=False):
            return True
        self._activity.release()
        return False

    def execution_in_progress(self, run_id: str) -> bool:
        record = self._runs.get(run_id)
        return record is not None and not record.ready and record.status in {"running", "working"}

    def start_execution(self, run_id: str) -> dict[str, Any] | None:
        """Start execution for an approved plan. A run already executing is left alone."""
        record = self._runs.get(run_id)
        if record is None:
            return None
        if self.agent_busy():
            return {"error": "The agent is still processing a request."}

        plan_storage = PlanStorageManager(self.config.runs_dir)
        plan = plan_storage.load_plan(run_id)
        if not plan or plan.status != "approved":
            return {"error": "Plan not found or not approved."}

        with self._lock:
            if self.execution_in_progress(run_id):
                return {"error": "An execution is already in progress."}
            record.generation += 1
            record.status = "running"
            record.ready = False
            record.steps = []
            record.execution_history = []
            record.snapshots = []
            record.report = None
            record.error = None
            record.control = RunControl()
            record.done = threading.Event()
        self._remember(record)

        thread = threading.Thread(target=self._execute, args=(run_id,), daemon=True)
        thread.start()
        return self.snapshot(record)

    def discuss_plan(self, run_id: str, message: str) -> dict[str, Any] | None:
        """Answer a question or update the plan. Holds the agent until the reply is saved."""
        record = self._runs.get(run_id)
        if record is None:
            return None
        text = message.strip()
        if not text:
            return {"error": "Message is required."}
        if not self._activity.acquire(blocking=False):
            raise AgentBusyError()
        try:
            plan_storage = PlanStorageManager(self.config.runs_dir)
            chat = plan_storage.load_chat_history(run_id) or PlannerChatHistory(run_id=run_id)
            chat.add_message("user", text)
            plan = plan_storage.load_plan(run_id)
            if plan is None:
                chat.add_message("assistant", "No plan exists yet. Please create a run first.")
            else:
                planner = self._planner()
                if not hasattr(planner, "refine_plan"):
                    chat.add_message("assistant", "Planner does not support plan refinement.")
                else:
                    history = [item.model_dump(mode="json") for item in chat.messages]
                    refined_phases, refined_steps, reasoning = planner.refine_plan(
                        plan.phases, plan.steps, text, history
                    )
                    changed = [phase.model_dump() for phase in refined_phases] != [
                        phase.model_dump() for phase in plan.phases
                    ] or [step.model_dump() for step in refined_steps] != [
                        step.model_dump() for step in plan.steps
                    ]
                    if changed:
                        plan.phases = refined_phases
                        plan.steps = refined_steps
                        plan.status = "draft"
                        plan_storage.save_plan(plan)
                        chat.add_message("assistant", f"Plan updated. {reasoning}")
                    else:
                        chat.add_message("assistant", reasoning)
            plan_storage.save_chat_history(chat)
            return {"messages": [item.model_dump(mode="json") for item in chat.messages]}
        finally:
            self._activity.release()

    def wait(self, run_id: str, timeout: float | None = None) -> dict[str, Any]:
        record = self._runs[run_id]
        record.done.wait(timeout)
        return self.snapshot(record)

    def iter_snapshots(self, run_id: str):
        record = self._runs.get(run_id)
        if record is None:
            return None
        index = 0
        while True:
            with self._lock:
                pending = record.snapshots[index:]
            if pending:
                for snap in pending:
                    index += 1
                    yield snap
                    if snap["ready"]:
                        return
                continue
            if record.done.is_set():
                return
            record.done.wait(timeout=0.05)

    def snapshot(self, record: RunRecord) -> dict[str, Any]:
        payload = {
            "id": record.id,
            "status": record.status,
            "ready": record.ready,
            "specification": record.specification,
            "steps": record.steps,
            "report": record.report,
            "execution_history": record.execution_history,
        }
        plan = PlanStorageManager(self.config.runs_dir).load_plan(record.id)
        if plan is not None:
            payload["plan"] = plan.model_dump(mode="json")
        return payload

    def run_dir(self, run_id: str) -> Path:
        return self.config.runs_dir / run_id

    def _validate(self, specification: str) -> None:
        if specification is None or not str(specification).strip():
            raise SpecValidationError("The specification is missing or blank.")
        size = len(specification.encode("utf-8"))
        if size > self.config.spec_max_bytes:
            raise SpecValidationError("The specification is larger than 100 KB.")

    def _execute(self, run_id: str) -> None:
        record = self._runs[run_id]
        generation = record.generation
        done = record.done
        record.status = "working"
        self._remember(record)
        if self.pause is not None:
            while not self.pause.is_set():
                if record.generation != generation or record.control.cancel_requested:
                    break
                self.pause.wait(timeout=0.05)
        if record.generation != generation:
            done.set()
            return
        if record.control.cancel_requested:
            record.status = "canceled"
            record.ready = True
            record.report = {
                "schema_version": "1",
                "id": run_id,
                "verdict": "canceled",
                "specification": record.specification,
                "reason_code": "canceled",
                "reason": "The run was canceled.",
                "missing": [],
                "steps": record.steps,
            }
            self._write_report(record)
            self._remember(record)
            done.set()
            return
        gui = None

        def publish(state: AgentState) -> None:
            report = state.get("report")
            if report:
                record.report = report
                record.status = status_for_verdict(report["verdict"])
                record.ready = record.status in TERMINAL_STATUSES
                record.steps = report["steps"]
                self._write_report(record)
            else:
                record.status = "working"
                record.ready = False
                record.steps = list(state.get("step_views") or [])
            record.execution_history = list(state.get("execution_history") or [])
            self._remember(record)

        gui = None
        coding = None
        try:
            gui = self.gui or build_gui(self.config)
            coding = self.coding or build_coding_agent(self.config.runs_dir, self.config.pi_llm_model)
            evidence = self.run_dir(run_id) / "evidence"
            evidence.mkdir(parents=True, exist_ok=True)
            work_dir = self.run_dir(run_id) / "work"
            work_dir.mkdir(parents=True, exist_ok=True)

            deps = GraphDeps(
                config=self.config,
                planner=self._planner(),
                gui=gui,
                cli=CLISubsystem(self.planner, work_dir),
                coding=coding,
                control=record.control,
                probe=self.probe,
                evidence_dir_for=lambda _run_id: str(evidence),
                judge=self.judge,
                chat_model=None,
                run_dir=self.run_dir(run_id),
                work_dir=work_dir,
            )

            state = initial_state(record.id, record.specification)
            state["max_retries"] = self.config.max_retries
            
            # Check if there's an approved plan to use
            plan_storage = PlanStorageManager(self.config.runs_dir)
            plan = plan_storage.load_plan(run_id)
            
            if plan and plan.status == "approved":
                # Use the approved plan, skip plan generation
                print(f"[DEBUG] Using approved plan for {run_id}: {len(plan.phases)} phases, {len(plan.steps)} steps")
                # Convert steps to the format expected by the graph
                step_dicts = [step.model_dump() for step in plan.steps]
                print(f"[DEBUG] Step dicts: {step_dicts[:1] if step_dicts else 'empty'}")
                state["test_matrix"] = step_dicts
                state["step_views"] = step_dicts
                state["current_step"] = 0  # Initialize current_step
                state["phase"] = "preflight"  # Skip to preflight
            else:
                # No approved plan, generate normally
                print(f"[DEBUG] No approved plan found, generating normally")
                pass
            
            print(f"[DEBUG] Starting graph execution with phase: {state['phase']}")
            print(f"[DEBUG] test_matrix type: {type(state.get('test_matrix'))}, length: {len(state.get('test_matrix', []))}")
            print(f"[DEBUG] step_views type: {type(state.get('step_views'))}, length: {len(state.get('step_views', []))}")
            try:
                state = run_graph(deps, state, publish)
                print(f"[DEBUG] Graph execution completed")
            except Exception as e:
                print(f"[DEBUG] Graph execution failed: {e}")
                import traceback
                traceback.print_exc()
                raise
                    
        except Exception as exc:  # noqa: BLE001
            print(f"[DEBUG] Execution failed with error: {exc}")
            import traceback
            traceback.print_exc()
            record.status = "failed"
            record.ready = True
            record.error = str(exc)
            record.report = {
                "schema_version": "1",
                "id": run_id,
                "verdict": "error",
                "specification": record.specification,
                "reason_code": "engine_error",
                "reason": str(exc),
                "missing": [],
                "steps": record.steps,
            }
            self._write_report(record)
            self._remember(record)
        finally:
            print(f"[DEBUG] In finally block")
            try:
                if gui is not None:
                    print(f"[DEBUG] Closing GUI subsystem")
                    closer = getattr(gui, "close", None)
                    if closer:
                        closer()
            except Exception as e:  # noqa: BLE001
                print(f"[DEBUG] Failed to close GUI subsystem: {e}")
                logger.warning("Failed to close GUI subsystem")
            try:
                if coding is not None:
                    print(f"[DEBUG] Closing coding subsystem")
                    coding_closer = getattr(coding, "close", None)
                    if coding_closer:
                        coding_closer()
            except Exception as e:  # noqa: BLE001
                print(f"[DEBUG] Failed to close coding subsystem: {e}")
            print(f"[DEBUG] Finally block completed")
            if record.generation != generation:
                done.set()
                return
            if not record.ready:
                record.ready = True
                record.status = record.status if record.status in TERMINAL_STATUSES else "failed"
                self._remember(record)
            print(f"[DEBUG] _execute method completed for {run_id}")
            done.set()

    def _remember(self, record: RunRecord) -> None:
        snap = self.snapshot(record)
        with self._lock:
            record.snapshots.append(snap)
        self._write_snapshot(record, snap)

    def _write_snapshot(self, record: RunRecord, snap: dict[str, Any]) -> None:
        path = self.run_dir(record.id) / "snapshot.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snap, indent=2), encoding="utf-8")

    def _load_saved_runs(self) -> None:
        if not self.config.runs_dir.is_dir():
            return
        found: list[tuple[float, dict[str, Any]]] = []
        for run_dir in self.config.runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            try:
                data = self._saved_run(run_dir)
            except Exception:  # noqa: BLE001 - one bad run must not hide the others
                logger.exception("Could not restore %s", run_dir.name)
                continue
            if data is None:
                continue
            found.append((run_dir.stat().st_mtime, data))
        for _, data in sorted(found, key=lambda item: item[0]):
            self._adopt_saved_run(data)

    def _saved_run(self, run_dir: Path) -> dict[str, Any] | None:
        snapshot_path = run_dir / "snapshot.json"
        if snapshot_path.is_file():
            try:
                data = json.loads(snapshot_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("Could not read %s", snapshot_path)
                return None
            if isinstance(data, dict) and data.get("id"):
                return data
        report_path = run_dir / "report.json"
        if not report_path.is_file():
            return None
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(report, dict):
            return None
        verdict = report.get("verdict") or "error"
        return {
            "id": report.get("id") or run_dir.name,
            "status": status_for_verdict(verdict),
            "ready": True,
            "specification": report.get("specification") or "",
            "steps": report.get("steps") or [],
            "report": report,
            "execution_history": [],
        }

    def _adopt_saved_run(self, data: dict[str, Any]) -> None:
        status = str(data.get("status") or "submitted")
        ready = bool(data.get("ready"))
        report = data.get("report")
        interrupted = status in {"running", "working"}
        if interrupted:
            status = "failed"
            ready = True
            report = {
                "schema_version": "1",
                "id": data.get("id"),
                "verdict": "error",
                "specification": data.get("specification") or "",
                "reason_code": "engine_error",
                "reason": "The execution stopped before it finished.",
                "missing": [],
                "steps": data.get("steps") or [],
            }
        record = RunRecord(
            id=str(data["id"]),
            specification=str(data.get("specification") or ""),
            status=status,
            ready=ready,
            steps=list(data.get("steps") or []),
            report=report if isinstance(report, dict) else None,
            execution_history=list(data.get("execution_history") or []),
        )
        record.done.set()
        self._runs[record.id] = record
        if interrupted:
            self._write_snapshot(record, self.snapshot(record))
            self._write_report(record)

    def _write_report(self, record: RunRecord) -> None:
        if record.report is None:
            return
        path = self.run_dir(record.id) / "report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record.report, indent=2), encoding="utf-8")


Runner = Callable[[], RunService]
