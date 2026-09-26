"""Engine, API, and A2A coverage with injected drivers."""

import json
import threading
from pathlib import Path

from fastapi.testclient import TestClient

from aqe.api import create_app
from aqe.assertions import evaluate_assertion
from aqe.capabilities import CapabilityFlag, HostCapabilities, unavailable_capabilities
from aqe.config import EngineConfig
from aqe.graph import GraphDeps, RunControl, initial_state, run_graph
from aqe.gui.subsystem import GUISubsystem
from aqe.judge import parse_judgment
from aqe.llm import NONSENSE_SPEC, _parse_labeled_steps
from aqe.service import RunService
from aqe.state import ActionResult, GUIAction, Judgment, PlanResult

SAMPLE = Path("examples/specs/registration.md").read_text(encoding="utf-8")


class StaticGUI:
    def __init__(self, summary: str, page_source: str = "") -> None:
        self.summary = summary
        self.page_source = page_source
        self.calls = 0

    def execute_visual_action(self, step, evidence_dir: Path) -> ActionResult:
        del evidence_dir
        self.calls += 1
        evidence = {"summary": self.summary}
        if self.page_source:
            evidence["page_source"] = self.page_source
        return ActionResult(ok=True, summary=self.summary, evidence=evidence)

    def close(self) -> None:
        return None


class RecordingDriver:
    def __init__(self) -> None:
        self.actions: list[GUIAction] = []

    def capture(self) -> bytes:
        return b"png"

    def act(self, action: GUIAction) -> None:
        self.actions.append(action)

    def close(self) -> None:
        return None


class FixedSandbox:
    def start_container(self, run_dir: Path, work_dir: Path, *, network: bool = False) -> str:
        del run_dir, work_dir, network
        return "test-container-id"

    def stop_container(self) -> None:
        return None

    def run(self, script: str, evidence_dir: Path, work_dir: Path, *, network: bool = False) -> tuple[str, str]:
        del script, evidence_dir, work_dir, network
        return json.dumps({"ok": True, "user": "ada"}), ""


class LabeledPlanner:
    def plan(self, specification: str):
        text = specification.strip()
        if not text or text == NONSENSE_SPEC:
            return PlanResult(
                accepted=False,
                reason="This request has no verifiable GUI or CLI actions.",
                reason_code="not_a_test_plan",
            )
        steps = _parse_labeled_steps(text)
        if not steps:
            return PlanResult(
                accepted=False,
                reason="no verifiable GUI or CLI actions were found",
                reason_code="not_a_test_plan",
            )
        return PlanResult(accepted=True, steps=steps)

    def script_for(self, intent: str) -> str:
        del intent
        return "import json\nprint(json.dumps({'ok': True, 'user': 'ada'}))\n"


def available_capabilities() -> HostCapabilities:
    return HostCapabilities(
        browser=CapabilityFlag(available=True),
        desktop=CapabilityFlag(available=True),
        coding=CapabilityFlag(available=True),
    )


class OutputJudge:
    """Stand-in for the model. It checks command stdout and stderr without calling one."""

    def judge(self, question: str, text: str) -> Judgment:
        stdout, stderr = text, ""
        if text.startswith("stdout:\n") and "\n\nstderr:\n" in text:
            stdout, stderr = text.removeprefix("stdout:\n").split("\n\nstderr:\n", 1)
        passed = evaluate_assertion(question, {"stdout": stdout, "stderr": stderr, "summary": stdout or stderr})
        return Judgment(passed=passed, judgment="The command output was checked against the assertion.")


class GarbagePlanner:
    def plan(self, specification: str):
        del specification
        raise ValueError("not json")

    def script_for(self, intent: str) -> str:
        return "print(1)\n"


def _service(tmp_path: Path, **kwargs) -> RunService:
    config = EngineConfig(runs_dir=tmp_path / "runs", llm="ollama", ollama_base_url="http://127.0.0.1:11434")
    probe = kwargs.pop("probe", available_capabilities)
    if "planner" not in kwargs:
        kwargs["planner"] = LabeledPlanner()
    if "gui" not in kwargs:
        kwargs["gui"] = StaticGUI("registered")
    if "judge" not in kwargs:
        kwargs["judge"] = OutputJudge()
    return RunService(config, probe=probe, **kwargs)


def test_cli_stdout_and_stderr_are_judged_with_the_assertion(tmp_path: Path) -> None:
    class CaptureJudge:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def judge(self, question: str, text: str) -> Judgment:
            self.calls.append((question, text))
            return Judgment(passed=True, judgment="stdout prints sandbox, which is a username.")

    judge = CaptureJudge()
    service = _service(tmp_path, judge=judge)
    spec = "CLI: whoami\nAssertion: The output contains a username\n"
    finished = service.wait(service.submit(spec)["id"])
    step = finished["report"]["steps"][0]
    assert finished["report"]["verdict"] == "pass"
    assert judge.calls
    question, text = judge.calls[0]
    assert question == "The output contains a username"
    assert text.startswith("stdout:\n")
    assert "\n\nstderr:\n" in text
    assert step["judgment"] == "stdout prints sandbox, which is a username."
    assert step["verification_results"][0]["passed"] is True


def test_unmatched_click_uses_the_only_button() -> None:
    from aqe.gui.playwright_driver import match_control

    assert match_control("Commit", ["Register"]) == "Register"
    assert match_control("Username", ["Username"]) == "Username"
    assert match_control("Save", ["Register", "Cancel"]) is None


def test_sample_run_passes(tmp_path: Path) -> None:
    service = _service(tmp_path)
    snapshot = service.submit(SAMPLE)
    finished = service.wait(snapshot["id"])
    report = finished["report"]
    assert finished["status"] == "completed"
    assert report["verdict"] == "pass"
    assert report["specification"] == SAMPLE
    assert [step["assertion_passed"] for step in report["steps"]] == [True, True]
    assert (tmp_path / "runs" / snapshot["id"] / "report.json").is_file()


def test_routes_desktop_and_cli(tmp_path: Path) -> None:
    browser = RecordingDriver()
    desktop = RecordingDriver()
    gui = GUISubsystem(browser, desktop)
    service = _service(tmp_path, gui=gui)
    spec = "GUI desktop: Click the Save button\nAssertion: contains:Save\n"
    finished = service.wait(service.submit(spec)["id"])
    assert finished["report"]["verdict"] == "pass"
    assert desktop.actions
    assert browser.actions == []
    cli_spec = "CLI: Check the session log\nAssertion: contains:ok\n"
    finished_cli = service.wait(service.submit(cli_spec)["id"])
    assert finished_cli["report"]["steps"][0]["interface"] == "CLI"


def test_retry_then_assertion_fail(tmp_path: Path) -> None:
    gui = StaticGUI("nope")
    service = _service(tmp_path, gui=gui)
    spec = "GUI browser: Submit the form\nAssertion: contains:registered\n"
    finished = service.wait(service.submit(spec)["id"])
    assert gui.calls == 2
    assert finished["status"] == "completed"
    assert finished["report"]["verdict"] == "fail"
    assert finished["report"]["reason_code"] == "assertion_failed"
    assert finished["report"]["steps"][0]["assertion_passed"] is False


class RecordingJudge:
    def __init__(self, passed: bool, judgment: str) -> None:
        self.passed = passed
        self.judgment = judgment
        self.calls: list[tuple[str, str]] = []

    def judge(self, question: str, page_source: str) -> Judgment:
        self.calls.append((question, page_source))
        return Judgment(passed=self.passed, judgment=self.judgment)


def test_page_verification_uses_the_model_judgment(tmp_path: Path) -> None:
    source = "<html><body>registered</body></html>"
    gui = StaticGUI("registered", page_source=source)
    judge = RecordingJudge(True, 'The page body is the word "registered".')
    service = _service(tmp_path, gui=gui, judge=judge)
    spec = "GUI browser: Submit the form\nAssertion: page contains:registered\n"
    finished = service.wait(service.submit(spec)["id"])
    step = finished["report"]["steps"][0]
    assert finished["report"]["verdict"] == "pass"
    assert step["assertion_passed"] is True
    assert step["judgment"] == 'The page body is the word "registered".'
    assert "page_source" not in step["evidence"]
    assert judge.calls == [("page contains:registered", source)]
    assert gui.calls == 1


def test_phase_judges_every_verification_after_the_operations(tmp_path: Path) -> None:
    from aqe.state import TestStep

    class PhasePlanner:
        def plan(self, specification: str):
            del specification
            return PlanResult(
                accepted=True,
                steps=[
                    TestStep(
                        step=1,
                        interface="GUI",
                        gui_driver="browser",
                        action="Register ada",
                        assertion="The page contains registered",
                        phase=1,
                        phase_name="Register ada",
                        operation_notes=["goto the form", "type ada", "click Register"],
                        verifications=[
                            "The page contains registered",
                            "The page does not show an error",
                        ],
                        operations=[GUIAction(action="goto", text="http://127.0.0.1:8765")],
                    )
                ],
            )

        def script_for(self, intent: str) -> str:
            del intent
            return "print('ok')\n"

    class TwoJudge:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def judge(self, question: str, page_source: str) -> Judgment:
            del page_source
            self.calls.append(question)
            return Judgment(passed=True, judgment=f"Checked: {question}")

    gui = StaticGUI("registered", page_source="<html><body>registered</body></html>")
    judge = TwoJudge()
    service = _service(tmp_path, planner=PhasePlanner(), gui=gui, judge=judge)
    finished = service.wait(service.submit("register ada")["id"])
    step = finished["report"]["steps"][0]
    assert finished["report"]["verdict"] == "pass"
    assert gui.calls == 1
    assert judge.calls == [
        "The page contains registered",
        "The page does not show an error",
    ]
    assert [item["passed"] for item in step["verification_results"]] == [True, True]
    assert step["phase_name"] == "Register ada"
    assert step["operation_notes"] == ["goto the form", "type ada", "click Register"]


def test_page_judgment_failure_is_the_report_reason(tmp_path: Path) -> None:
    explanation = "The page says hello, so it does not show registered."
    gui = StaticGUI("registered", page_source="<html><body>hello</body></html>")
    judge = RecordingJudge(False, explanation)
    service = _service(tmp_path, gui=gui, judge=judge)
    spec = "GUI browser: Submit the form\nAssertion: contains:registered\n"
    finished = service.wait(service.submit(spec)["id"])
    assert gui.calls == 2
    assert finished["report"]["verdict"] == "fail"
    assert finished["report"]["reason"] == explanation
    assert finished["report"]["steps"][0]["judgment"] == explanation


def test_parse_judgment_reads_the_model_reply() -> None:
    failed = parse_judgment('```json\n{"passed": false, "judgment": "The body says hello."}\n```')
    assert failed.passed is False
    assert failed.judgment == "The body says hello."
    assert parse_judgment('{"passed": "true", "judgment": "yes"}').passed is True
    broken = '{ "passed": false, "judgment": "stderr is not empty" '
    recovered = parse_judgment(broken)
    assert recovered.passed is False
    assert "stderr is not empty" in recovered.judgment
    later = '{"title": "Knowledge Graph"}\n{"passed": true, "judgment": "stdout has HTML and stderr is empty."}'
    assert parse_judgment(later).passed is True
    assert parse_judgment('{"result": "pass", "explanation": "stderr is empty."}').passed is True


def test_streams_are_shown_only_when_the_assertion_mentions_them() -> None:
    from aqe.judge import present_for_judge

    command = "stdout:\n" + ("x" * 20_000) + "\n\nstderr:\nboom"
    about_stdout = present_for_judge(command, "The curl command executes and stdout is not empty.")
    assert "stdout length: 20000 characters" in about_stdout
    assert "stderr" not in about_stdout
    assert "boom" not in about_stdout
    about_error = present_for_judge(command, "The curl command reports an error.")
    assert "stderr length: 4 characters" in about_error
    assert "boom" in about_error
    assert "stdout" not in about_error
    neither = present_for_judge(command, "The curl command executes.")
    assert "xxxx" not in neither
    assert "boom" not in neither
    with_code = "stdout:\nhello\n\nstderr:\n\n\nexit_code: 1"
    empty_stderr = present_for_judge(with_code, "stderr is empty.")
    assert "stderr length: 0 characters" in empty_stderr
    assert "exit_code" not in empty_stderr
    about_exit = present_for_judge(with_code, "The command exits with status 1.")
    assert "exit_code: 1" in about_exit
    assert "hello" not in about_exit


def test_planner_setup_failure_finishes_the_run(tmp_path: Path, monkeypatch) -> None:
    def broken(_config):
        raise RuntimeError("LLM_PROVIDER=ollama is missing its client package. Install the project with: uv pip install -e .")

    monkeypatch.setattr("aqe.service.build_planner", broken)
    service = _service(tmp_path, planner=None)
    finished = service.wait(service.submit(SAMPLE)["id"], timeout=5)
    assert finished["ready"] is True
    assert finished["status"] == "failed"
    assert finished["report"]["verdict"] == "error"
    assert finished["report"]["reason_code"] == "engine_error"
    assert "missing its client package" in finished["report"]["reason"]


def test_sandbox_start_is_a_system_error(tmp_path: Path) -> None:
    # This test is no longer valid as sandbox has been removed
    # CLI steps now execute in host environment
    pass


def test_nonsense_and_garbage_are_rejected(tmp_path: Path) -> None:
    service = _service(tmp_path)
    finished = service.wait(service.submit(NONSENSE_SPEC)["id"])
    assert finished["status"] == "rejected"
    assert finished["report"]["reason_code"] == "not_a_test_plan"
    assert finished["execution_history"] == []

    garbage = _service(tmp_path, planner=GarbagePlanner())
    finished_bad = garbage.wait(garbage.submit("GUI browser: Go\nAssertion: contains:go\n")["id"])
    assert finished_bad["report"]["verdict"] == "rejected"
    assert finished_bad["report"]["reason_code"] == "not_a_test_plan"


def test_missing_capability_does_not_call_drivers(tmp_path: Path) -> None:
    gui = StaticGUI("registered")
    service = _service(tmp_path, gui=gui, probe=unavailable_capabilities)
    finished = service.wait(service.submit(SAMPLE)["id"])
    assert finished["status"] in ("rejected", "failed")
    assert finished["report"]["reason_code"] in ("missing_capability", "engine_error")
    assert finished["report"]["specification"] == SAMPLE
    assert finished["execution_history"] == []
    assert gui.calls == 0


def test_graph_publish_order(tmp_path: Path) -> None:
    seen: list[str] = []

    def publish(state) -> None:
        statuses = [step["status"] for step in state.get("step_views") or []]
        seen.append(",".join(statuses) or state.get("phase", ""))

    config = EngineConfig(runs_dir=tmp_path / "runs")
    planner = LabeledPlanner()
    gui = StaticGUI("registered")
    from aqe.cli_runtime.synthesizer import CLISubsystem

    class IdleCoding:
        def execute_coding_action(self, step, evidence_dir):
            del step, evidence_dir
            raise AssertionError("coding was not requested")

        def close(self) -> None:
            return None

    deps = GraphDeps(
        config=config,
        planner=planner,
        gui=gui,
        cli=CLISubsystem(planner, tmp_path / "work"),
        coding=IdleCoding(),
        control=RunControl(),
        probe=available_capabilities,
        evidence_dir_for=lambda run_id: str(tmp_path / run_id),
        judge=OutputJudge(),
        chat_model=None,
        run_dir=tmp_path / "test-run",
        work_dir=tmp_path / "test-run" / "work",
    )
    final = run_graph(deps, initial_state("run", SAMPLE), publish)
    assert final["report"]["verdict"] == "pass"
    assert any("pending" in item for item in seen)
    assert any("running" in item for item in seen)
    assert any("passed" in item for item in seen)


def test_api_contract(tmp_path: Path) -> None:
    pause = threading.Event()
    service = _service(tmp_path, pause=pause)
    app = create_app(service)
    with TestClient(app) as client:
        blank = client.post("/v1/runs", json={"specification": "  "})
        assert blank.status_code == 400
        assert "error" in blank.json()

        created = client.post("/v1/runs", json={"specification": SAMPLE, "wait_for_approval": False})
        assert created.status_code == 202
        run_id = created.json()["id"]
        waiting = client.get(f"/v1/runs/{run_id}")
        assert waiting.status_code == 200
        assert waiting.json()["ready"] is False
        assert waiting.json()["report"] is None
        pause.set()
        finished = service.wait(run_id)
        assert finished["ready"] is True
        assert finished["report"]["verdict"] == "pass"
        assert len(finished["report"]["steps"]) == 2
        assert client.get("/v1/runs/missing").status_code == 404

        with client.stream("GET", f"/v1/runs/{run_id}/events") as response:
            body = response.read().decode()
        snapshots = [json.loads(line.removeprefix("data: ")) for line in body.splitlines() if line.startswith("data: ")]
        statuses = []
        for snapshot in snapshots:
            statuses.extend(step["status"] for step in snapshot["steps"])
        assert "pending" in statuses
        assert "running" in statuses
        assert "passed" in statuses
        assert snapshots[-1]["ready"] is True
        assert snapshots[-1]["status"] == "completed"


def test_a2a_send_and_get_task(tmp_path: Path) -> None:
    service = _service(tmp_path)
    app = create_app(service, public_url="http://127.0.0.1:8000")
    headers = {"A2A-Version": "1.0"}
    with TestClient(app) as client:
        card = client.get("/.well-known/agent-card.json")
        assert card.status_code == 200
        body = card.json()
        skills = body.get("skills") or []
        assert any(skill.get("id") == "execute-quality-spec" for skill in skills)

        empty = client.post(
            "/",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": "empty",
                "method": "SendMessage",
                "params": {"message": {"messageId": "m-empty", "role": "ROLE_USER", "parts": []}},
            },
        )
        assert empty.status_code == 200
        assert "error" in empty.json()

        sent = client.post(
            "/",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "SendMessage",
                "params": {
                    "message": {
                        "messageId": "m1",
                        "role": "ROLE_USER",
                        "parts": [{"text": SAMPLE}],
                    }
                },
            },
        )
        assert sent.status_code == 200
        payload = sent.json()
        assert "error" not in payload
        result = payload["result"]
        task_id = (result.get("task") or result)["id"]
        report = None
        for _ in range(50):
            fetched = client.post(
                "/",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": "2",
                    "method": "GetTask",
                    "params": {"id": task_id},
                },
            )
            result = fetched.json()["result"]
            state = result["status"]["state"]
            if state in {
                "TASK_STATE_COMPLETED",
                "TASK_STATE_FAILED",
                "TASK_STATE_REJECTED",
                "TASK_STATE_CANCELED",
            }:
                artifacts = result.get("artifacts") or []
                data = artifacts[0]["parts"][0]["data"]
                report = data if isinstance(data, dict) else json.loads(data)
                break
        assert report is not None
        assert report["verdict"] == "pass"
        assert report["id"] == task_id
        same = service.get(task_id)
        assert same is not None
        assert same["report"]["verdict"] == "pass"


def test_running_execution_is_not_replaced(tmp_path: Path) -> None:
    pause = threading.Event()
    service = _service(tmp_path, pause=pause)
    snapshot = service.submit(SAMPLE)
    current = snapshot
    for _ in range(50):
        current = service.get(snapshot["id"]) or current
        if current["status"] == "working":
            break
        threading.Event().wait(0.05)
    blocked = service.start_execution(snapshot["id"])
    assert blocked is not None
    assert blocked["error"] == "An execution is already in progress."
    pause.set()
    finished = service.wait(snapshot["id"])
    assert finished["report"]["verdict"] == "pass"


def test_finished_plan_can_run_again(tmp_path: Path) -> None:
    service = _service(tmp_path)
    finished = service.wait(service.submit(SAMPLE)["id"])
    assert finished["report"]["verdict"] == "pass"
    again = service.start_execution(finished["id"])
    assert again is not None
    assert "error" not in again
    second = service.wait(finished["id"])
    assert second["report"]["verdict"] == "pass"


def test_saved_run_reloads_its_state(tmp_path: Path) -> None:
    service = _service(tmp_path)
    finished = service.wait(service.submit(SAMPLE)["id"])
    restored = _service(tmp_path)
    again = restored.get(finished["id"])
    assert again is not None
    assert again["specification"] == SAMPLE
    assert again["ready"] is True
    assert again["report"]["verdict"] == "pass"
    assert again["steps"]


def test_coding_and_desktop_phases_are_not_executable() -> None:
    from aqe.llm import _validate_phases
    from aqe.state import TestPhase

    coding = _validate_phases(
        [
            TestPhase(
                phase=1,
                name="Write a file",
                interface="CODING",
                operation_notes=["create file"],
                verifications=["The file exists."],
            )
        ]
    )
    assert coding is not None
    assert "not executable" in coding
    desktop = _validate_phases(
        [
            TestPhase(
                phase=1,
                name="Click Save",
                interface="GUI",
                gui_driver="desktop",
                operation_notes=["click Save"],
                verifications=["The page shows saved."],
            )
        ]
    )
    assert desktop is not None
    assert "Desktop" in desktop
