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
from aqe.errors import SpecValidationError
from aqe.graph import GraphDeps, RunControl, initial_state, run_graph
from aqe.gui.subsystem import GUISubsystem, build_gui
from aqe.plan_storage import PlanStorageManager
from aqe.judge import PageJudge
from aqe.llm import Planner, build_planner
from aqe.plan_storage import PlanStorageManager
from aqe.state import AgentState, PlanStorage, TERMINAL_STATUSES, status_for_verdict

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
        self.planner = planner or build_planner(self.config)
        self.gui = gui
        self.coding = coding
        self.judge = judge
        self.probe = probe or (lambda: probe_host(self.config))
        self.pause = pause
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    def submit(self, specification: str, *, run_id: str | None = None, wait_for_approval: bool = True) -> dict[str, Any]:
        self._validate(specification)
        record = RunRecord(
            id=run_id or uuid.uuid4().hex,
            specification=specification,
        )
        with self._lock:
            self._runs[record.id] = record
        self._remember(record)
        
        # Generate initial plan and store it
        plan_result = self.planner.plan(specification)
        
        if plan_result.accepted:
            # Create plan storage
            plan_storage = PlanStorageManager(self.config.runs_dir)
            plan = PlanStorage(
                run_id=record.id,
                status="draft" if wait_for_approval else "approved",
                phases=plan_result.phases,
                steps=plan_result.steps,
            )
            plan_storage.save_plan(plan)
            
            # If waiting for approval, don't start execution yet
            if wait_for_approval:
                record.status = "submitted"
                self._remember(record)
                return self.snapshot(record)
        
        # Start execution immediately if not waiting for approval or plan was rejected
        thread = threading.Thread(target=self._execute, args=(record.id,), daemon=True)
        thread.start()
        return self.snapshot(record)

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

    def start_execution(self, run_id: str) -> dict[str, Any] | None:
        """Start execution for a run that was waiting for plan approval or re-trigger execution for a completed/failed run."""
        print(f"[DEBUG] start_execution called for {run_id}")
        record = self._runs.get(run_id)
        if record is None:
            print(f"[DEBUG] Record not found for {run_id}")
            return None
        
        print(f"[DEBUG] Record found, status: {record.status}")
        
        # Check if plan is approved
        plan_storage = PlanStorageManager(self.config.runs_dir)
        plan = plan_storage.load_plan(run_id)
        if not plan or plan.status != "approved":
            print(f"[DEBUG] Plan not found or not approved: {plan.status if plan else 'None'}")
            return {"error": "Plan not found or not approved."}
        
        print(f"[DEBUG] Plan approved, checking record status")
        
        # Cancel any existing execution that is currently running
        if record.status in ["running", "working"]:
            print(f"[DEBUG] Canceling existing execution, status: {record.status}")
            record.control.cancel_requested = True
            # Wait a moment for cancellation to take effect
            import time
            time.sleep(0.5)
        
        print(f"[DEBUG] Resetting run record")
        # Reset the run record for new execution
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
        
        print(f"[DEBUG] Starting execution thread")
        # Start execution
        thread = threading.Thread(target=self._execute, args=(run_id,), daemon=True)
        thread.start()
        
        print(f"[DEBUG] Execution thread started, returning snapshot")
        return self.snapshot(record)

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
        return {
            "id": record.id,
            "status": record.status,
            "ready": record.ready,
            "specification": record.specification,
            "steps": record.steps,
            "report": record.report,
            "execution_history": record.execution_history,
        }

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
        record.status = "working"
        self._remember(record)
        if self.pause is not None:
            self.pause.wait()
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
                planner=self.planner,
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
            if not record.ready:
                record.ready = True
                record.status = record.status if record.status in TERMINAL_STATUSES else "failed"
                self._remember(record)
            print(f"[DEBUG] _execute method completed for {run_id}")
            record.done.set()

    def _remember(self, record: RunRecord) -> None:
        snap = self.snapshot(record)
        with self._lock:
            record.snapshots.append(snap)

    def _write_report(self, record: RunRecord) -> None:
        if record.report is None:
            return
        path = self.run_dir(record.id) / "report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record.report, indent=2), encoding="utf-8")


Runner = Callable[[], RunService]
