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
from aqe.cli_runtime.sandbox import build_sandbox
from aqe.cli_runtime.synthesizer import CLISubsystem
from aqe.coding_agent.subsystem import CodingSubsystem, build_coding_agent
from aqe.config import EngineConfig
from aqe.errors import HarnessError, SpecValidationError
from aqe.graph import GraphDeps, RunControl, initial_state, run_graph
from aqe.gui.subsystem import GUISubsystem, build_gui
from aqe.judge import PageJudge
from aqe.llm import Planner, build_planner
from aqe.state import TERMINAL_STATUSES, AgentState, status_for_verdict

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
        sandbox: Any | None = None,
        coding: CodingSubsystem | None = None,
        judge: PageJudge | None = None,
        probe: Probe | None = None,
        pause: threading.Event | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.config.runs_dir.mkdir(parents=True, exist_ok=True)
        self.planner = planner
        self.gui = gui
        self.sandbox = sandbox
        self.coding = coding
        self.judge = judge
        self.probe = probe or (lambda: probe_host(self.config))
        self.pause = pause
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    def submit(self, specification: str, *, run_id: str | None = None) -> dict[str, Any]:
        self._validate(specification)
        record = RunRecord(
            id=run_id or uuid.uuid4().hex,
            specification=specification,
        )
        with self._lock:
            self._runs[record.id] = record
        self._remember(record)
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
        sandbox = None
        try:
            config = self.config
            planner = self.planner or build_planner(config)
            gui = self.gui or build_gui(config)
            sandbox = self.sandbox or build_sandbox(config)
            coding = self.coding or build_coding_agent(config.runs_dir, config.pi_llm_model)
            evidence = self.run_dir(run_id) / "evidence"
            evidence.mkdir(parents=True, exist_ok=True)
            work_dir = self.run_dir(run_id) / "work"
            work_dir.mkdir(parents=True, exist_ok=True)

            deps = GraphDeps(
                config=config,
                planner=planner,
                gui=gui,
                cli=CLISubsystem(planner, sandbox),
                coding=coding,
                control=record.control,
                probe=self.probe,
                evidence_dir_for=lambda _run_id: str(evidence),
                judge=self.judge,
                chat_model=None,
                run_dir=self.run_dir(run_id),
                work_dir=work_dir,
            )

            # Check capabilities before starting graph to avoid sandbox start if unavailable
            capabilities = self.probe()
            if not capabilities.sandbox.available:
                # Skip sandbox start if unavailable
                deps.run_dir = None
                deps.work_dir = None
            else:
                # Start the shared sandbox container for this run
                try:
                    sandbox.start_container(self.run_dir(run_id), work_dir)
                except HarnessError as exc:
                    # Handle sandbox start failures specifically
                    record.status = "failed"
                    record.ready = True
                    record.error = str(exc)
                    record.report = {
                        "schema_version": "1",
                        "id": run_id,
                        "verdict": "error",
                        "specification": record.specification,
                        "reason_code": exc.code,
                        "reason": str(exc),
                        "missing": [],
                        "steps": record.steps,
                    }
                    self._write_report(record)
                    self._remember(record)
                    return

            run_graph(deps, initial_state(run_id, record.specification), publish)
        except Exception as exc:  # noqa: BLE001
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
            try:
                if gui is not None:
                    closer = getattr(gui, "close", None)
                    if closer:
                        closer()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to close GUI subsystem")
            try:
                if coding is not None:
                    coding_closer = getattr(coding, "close", None)
                    if coding_closer:
                        coding_closer()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to close coding subsystem")
            if not record.ready:
                record.ready = True
                record.status = record.status if record.status in TERMINAL_STATUSES else "failed"
                self._remember(record)
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
