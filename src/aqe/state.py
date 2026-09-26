"""Pydantic payloads and the LangGraph state dictionary."""

from datetime import UTC, datetime
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

Interface = Literal["GUI", "CLI", "CODING"]
GuiDriverName = Literal["browser", "desktop"]
StepStatus = Literal[
    "pending",
    "running",
    "passed",
    "failed",
    "retrying",
    "error",
    "skipped",
]
RunStatus = Literal[
    "submitted",
    "working",
    "completed",
    "failed",
    "rejected",
    "canceled",
]
Verdict = Literal["pass", "fail", "error", "rejected", "canceled"]
ReasonCode = Literal[
    "assertion_failed",
    "browser_launch_failed",
    "desktop_input_failed",
    "driver_timeout",
    "engine_error",
    "not_a_test_plan",
    "missing_capability",
    "canceled",
    "coding_agent_failed",
    "coding_agent_not_available",
]
PlanStatus = Literal["draft", "approved", "rejected", "executing", "completed"]


class GUIAction(BaseModel):
    """One pixel or locator action inside a GUI step."""

    action: Literal["click", "type", "press", "goto", "run"]
    coordinate: list[int] | None = None
    selector: dict[str, str] | None = None
    text: str | None = None


class CodingAction(BaseModel):
    """One coding operation for the Pi coding agent."""

    action: Literal["create_file", "update_file", "review_code", "execute_code", "run"]
    file_path: str | None = None
    content: str | None = None
    description: str | None = None
    command: str | None = None
    timeout: int | None = None


class TestPhase(BaseModel):
    """One pipeline stage: operations first, then the checks of those operations."""

    phase: int
    name: str
    interface: Interface
    gui_driver: GuiDriverName | None = None
    depends_on: list[int] = Field(default_factory=list)
    operations: list[GUIAction] = Field(default_factory=list)
    operation_notes: list[str] = Field(default_factory=list)
    verifications: list[str] = Field(default_factory=list)
    coding_operations: list[CodingAction] = Field(default_factory=list)

    def validate_shape(self) -> str | None:
        if self.interface == "GUI" and self.gui_driver == "desktop":
            return f"phase {self.phase} targets a desktop application, which is not tested"
        if self.interface == "CODING" and self.gui_driver is not None:
            return f"phase {self.phase} is a CODING phase with a gui_driver"
        if self.interface == "CODING" and not self.coding_operations:
            return f"phase {self.phase} is a CODING phase with no coding operations"
        return None


class TestStep(BaseModel):
    """One planned phase as the executor runs it."""

    step: int
    interface: Interface
    gui_driver: GuiDriverName | None = None
    action: str
    assertion: str | None = ""  # Made optional for non-testing steps
    operations: list[GUIAction] = Field(default_factory=list)
    phase: int | None = None
    phase_name: str | None = None
    depends_on: list[int] = Field(default_factory=list)
    operation_notes: list[str] = Field(default_factory=list)
    verifications: list[str] = Field(default_factory=list)
    coding_operations: list[CodingAction] = Field(default_factory=list)

    def validate_shape(self) -> str | None:
        if not self.action.strip():
            return f"step {self.step} has an empty action"
        # Assertion is now optional for non-testing steps
        # None or empty string is valid (non-testing step)
        if self.assertion and not self.assertion.strip():
            return f"step {self.step} has an empty assertion"
        if self.interface == "GUI" and self.gui_driver == "desktop":
            return f"step {self.step} targets a desktop application, which is not tested"
        if self.interface == "GUI" and self.gui_driver is None:
            return f"step {self.step} is a GUI step with no gui_driver"
        if self.interface == "CLI" and self.gui_driver is not None:
            return f"step {self.step} is a CLI step with a gui_driver"
        if self.interface == "CODING" and self.gui_driver is not None:
            return f"step {self.step} is a CODING step with a gui_driver"
        if self.interface == "CODING" and not self.coding_operations:
            return f"step {self.step} is a CODING step with no coding operations"
        return None


class Judgment(BaseModel):
    """An LLM decision about one verification question."""

    passed: bool
    judgment: str


class ActionResult(BaseModel):
    """What a driver returned after an action finished."""

    ok: bool
    summary: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    assertion_passed: bool | None = None


class VerificationResult(BaseModel):
    """One check inside a phase, and the judgment of that check."""

    question: str
    passed: bool | None = None
    judgment: str | None = None


class StepView(BaseModel):
    """A phase as shown on the live snapshot and in the report."""

    step: int
    interface: Interface
    gui_driver: GuiDriverName | None = None
    action: str
    assertion: str
    status: StepStatus = "pending"
    assertion_passed: bool | None = None
    judgment: str | None = None
    summary: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    phase: int | None = None
    phase_name: str | None = None
    depends_on: list[int] = Field(default_factory=list)
    operation_notes: list[str] = Field(default_factory=list)
    verification_results: list[VerificationResult] = Field(default_factory=list)
    coding_operations: list[CodingAction] = Field(default_factory=list)

    @classmethod
    def from_step(cls, step: TestStep, status: StepStatus = "pending") -> "StepView":
        questions = step.verifications or ([step.assertion] if step.assertion.strip() else [])
        return cls(
            step=step.step,
            interface=step.interface,
            gui_driver=step.gui_driver if step.interface != "CODING" else None,
            action=step.action,
            assertion=step.assertion,
            status=status,
            phase=step.phase,
            phase_name=step.phase_name,
            depends_on=list(step.depends_on),
            operation_notes=list(step.operation_notes),
            verification_results=[VerificationResult(question=question) for question in questions],
            coding_operations=list(step.coding_operations),
        )


class TestReport(BaseModel):
    """The only result document callers branch on."""

    schema_version: Literal["1"] = "1"
    id: str
    verdict: Verdict
    specification: str
    reason_code: ReasonCode | None = None
    reason: str | None = None
    missing: list[str] = Field(default_factory=list)
    steps: list[StepView] = Field(default_factory=list)


class PlanResult(BaseModel):
    """Planner output before any driver runs."""

    accepted: bool
    reason: str | None = None
    reason_code: ReasonCode | None = None
    phases: list[TestPhase] = Field(default_factory=list)
    steps: list[TestStep] = Field(default_factory=list)


class PlanStorage(BaseModel):
    """Stored plan with approval status."""

    run_id: str
    status: PlanStatus = "draft"
    phases: list[TestPhase] = Field(default_factory=list)
    steps: list[TestStep] = Field(default_factory=list)
    approved_at: datetime | None = None
    approved_by: str | None = None  # User identifier
    rejection_reason: str | None = None


class PlannerMessage(BaseModel):
    """A message in the planner chat."""

    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PlannerChatHistory(BaseModel):
    """Chat history for planner interaction."""

    run_id: str
    messages: list[PlannerMessage] = Field(default_factory=list)

    def add_message(self, role: Literal["user", "assistant"], content: str) -> None:
        """Add a message to the chat history."""
        self.messages.append(PlannerMessage(role=role, content=content))


class AgentState(TypedDict, total=False):
    """LangGraph state. Step payloads are stored as plain dicts."""

    run_id: str
    specification: str
    current_step: int
    test_matrix: list[dict[str, Any]]
    execution_history: list[dict[str, Any]]
    attempt_counts: dict[str, int]
    phase: str
    step_views: list[dict[str, Any]]
    max_retries: int
    last_result: dict[str, Any] | None
    report: dict[str, Any] | None
    reason_code: str | None
    reason: str | None
    missing: list[str]
    coding_instructions: list[dict[str, Any]]  # Track LLM's instructions to coding agent


def status_for_verdict(verdict: Verdict) -> RunStatus:
    if verdict in ("pass", "fail"):
        return "completed"
    if verdict == "error":
        return "failed"
    if verdict == "rejected":
        return "rejected"
    return "canceled"


TERMINAL_STATUSES = frozenset({"completed", "failed", "rejected", "canceled"})
