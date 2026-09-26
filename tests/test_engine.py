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
from aqe.plan_storage import PlanStorageManager
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
        coding=CapabilityFlag(available=True),
    )


class OutputJudge:
    """Stand-in for the model. It checks command stdout and stderr without calling one."""

    def judge(self, question: str, text: str) -> Judgment:
        stdout, stderr = text, ""
        if "\nstdout:\n" in text:
            stdout = text.split("\nstdout:\n", 1)[1]
        elif text.startswith("stdout:\n"):
            stdout = text.removeprefix("stdout:\n")
        if "\n\nstderr length:" in stdout:
            stdout = stdout.split("\n\nstderr length:", 1)[0]
        elif "\n\nstderr:\n" in stdout:
            stdout, stderr = stdout.split("\n\nstderr:\n", 1)
        if "\nstderr:\n" in text:
            stderr = text.split("\nstderr:\n", 1)[1].split("\n\n", 1)[0]
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
    assert "$?:" in text
    assert "stdout:" in text
    assert "stderr:" in text
    assert "Verification statement:" in text
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


def test_routes_browser_and_cli(tmp_path: Path) -> None:
    browser = RecordingDriver()
    gui = GUISubsystem(browser)
    service = _service(tmp_path, gui=gui)
    spec = "GUI browser: Click the Save button\nAssertion: contains:Save\n"
    finished = service.wait(service.submit(spec)["id"])
    assert finished["report"]["verdict"] == "pass"
    assert browser.actions
    cli_spec = "CLI: Check the session log\nAssertion: contains:ok\n"
    finished_cli = service.wait(service.submit(cli_spec)["id"])
    assert finished_cli["report"]["steps"][0]["interface"] == "CLI"


def test_cli_step_runs_its_bash_script(tmp_path: Path) -> None:
    from aqe.cli_runtime.synthesizer import CLISubsystem
    from aqe.state import TestStep

    class Planner:
        def script_for(self, intent: str) -> str:
            del intent
            raise AssertionError("a CLI step with a bash script does not ask for another script")

    step = TestStep(
        step=1,
        interface="CLI",
        action="Install dependencies",
        assertion="Dependencies are installed.",
        script="set -e\necho installed-in-venv\n",
        verifications=["Dependencies are installed."],
    )
    result = CLISubsystem(Planner(), tmp_path / "work").execute_runtime_action(step, tmp_path / "evidence")
    assert result.ok
    assert result.evidence["stdout"] == "installed-in-venv\n"
    assert (tmp_path / "work" / "script.sh").read_text(encoding="utf-8").startswith("set -e\n")
    assert "VIRTUAL_ENV=" in result.evidence["script_result"]


def test_bash_script_records_the_virtual_environment(tmp_path: Path) -> None:
    from aqe.cli_runtime.synthesizer import CLISubsystem
    from aqe.state import TestStep

    class Planner:
        def script_for(self, intent: str) -> str:
            del intent
            raise AssertionError("a CLI step with a bash script does not ask for another script")

    step = TestStep(
        step=1,
        interface="CLI",
        action="Create and activate virtual environment",
        assertion="A virtual environment named venv has been created and activated.",
        script="python3 -m venv venv\nsource venv/bin/activate\n",
        verifications=["A virtual environment named venv has been created and activated."],
    )
    result = CLISubsystem(Planner(), tmp_path / "work").execute_runtime_action(step, tmp_path / "evidence")
    assert result.ok
    assert result.evidence["stdout"] == ""
    recorded = result.evidence["script_result"]
    assert "venv_dir=present" in recorded
    assert "venv_python=present" in recorded
    assert f"VIRTUAL_ENV={tmp_path / 'work' / 'venv'}" in recorded


def test_a_cli_verification_includes_the_exit_code() -> None:
    from aqe.graph import validate_node
    from aqe.judge import Judgment
    from aqe.state import ActionResult, StepView, TestStep

    class Judge:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def judge(self, question: str, text: str) -> Judgment:
            self.calls.append((question, text))
            return Judgment(passed=False, judgment="$? is 1 and stderr says No such file.")

    step = TestStep(
        step=1,
        interface="CLI",
        action="Install dependencies",
        assertion="Dependencies are installed.",
        verifications=["Dependencies are installed."],
        script="pip install -r requirements.txt\n",
    )
    judge = Judge()
    deps = type("Deps", (), {"planner": object(), "judge": judge, "work_dir": None})()
    result = ActionResult(
        ok=False,
        summary="Bash script executed.",
        evidence={"stdout": "", "stderr": "No such file", "exit_code": 1, "execution_type": "host"},
    )
    update = validate_node(
        {
            "current_step": 0,
            "max_retries": 0,
            "test_matrix": [step.model_dump()],
            "step_views": [StepView.from_step(step).model_dump()],
            "last_result": result.model_dump(),
            "execution_history": [],
            "attempt_counts": {},
        },
        deps,
    )
    assert judge.calls
    question, text = judge.calls[0]
    assert question == "Dependencies are installed."
    assert "$?: 1" in text
    assert "stdout:" in text
    assert "No such file" in text
    assert "Verification statement:\nDependencies are installed." in text
    assert update["phase"] == "finish"
    assert "$? is 1" in update["reason"]


def test_a_zero_exit_is_judged_from_stdout_and_stderr() -> None:
    from aqe.graph import validate_node
    from aqe.judge import Judgment
    from aqe.state import ActionResult, StepView, TestStep

    class Judge:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def judge(self, question: str, text: str) -> Judgment:
            self.calls.append((question, text))
            return Judgment(passed=True, judgment="stdout shows the virtual environment.")

    step = TestStep(
        step=1,
        interface="CLI",
        action="Create and activate virtual environment",
        assertion="A virtual environment named venv has been created and activated.",
        verifications=["A virtual environment named venv has been created and activated."],
        script="python3 -m venv venv\nsource venv/bin/activate\n",
    )
    judge = Judge()
    deps = type("Deps", (), {"planner": object(), "judge": judge})()

    result = ActionResult(
        ok=True,
        summary="Bash script executed.",
        evidence={
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "script_result": "script_exit=0\nVIRTUAL_ENV=/work/venv\nvenv_dir=present\n",
            "execution_type": "host",
        },
    )
    update = validate_node(
        {
            "current_step": 0,
            "max_retries": 0,
            "test_matrix": [step.model_dump()],
            "step_views": [StepView.from_step(step).model_dump()],
            "last_result": result.model_dump(),
            "execution_history": [],
            "attempt_counts": {},
        },
        deps,
    )
    assert update["phase"] == "route"
    assert judge.calls
    text = judge.calls[0][1]
    assert "stdout:" in text
    assert "stderr:" in text
    assert "$?: 0" in text
    assert "Verification statement:\nA virtual environment named venv has been created and activated." in text


def test_a_cli_verification_runs_the_model_command(tmp_path: Path) -> None:
    from aqe.graph import validate_node
    from aqe.judge import CliCheck
    from aqe.state import ActionResult, StepView, TestStep

    statement = "The content of service.log is available for later inspection to confirm the background service is running."
    work = tmp_path / "work"
    work.mkdir()
    (work / "service.log").write_text("Running on http://0.0.0.0:8080\n", encoding="utf-8")

    class Judge:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, str, str, bool]] = []

        def judge_cli(self, question: str, exit_code: int, stdout: str, stderr: str, *, follow_up: bool = False) -> CliCheck:
            self.calls.append((question, exit_code, stdout, stderr, follow_up))
            if not follow_up:
                return CliCheck(
                    passed=None,
                    judgment="Read service.log to confirm the background service wrote it.",
                    script="cat service.log\n",
                )
            return CliCheck(passed=True, judgment="service.log contains Running on http://0.0.0.0:8080.")

    step = TestStep(
        step=1,
        interface="CLI",
        action="Start API Service",
        assertion=statement,
        verifications=[statement],
        script="python app.py > service.log 2>&1 &\n",
    )
    judge = Judge()
    deps = type("Deps", (), {"planner": object(), "judge": judge, "work_dir": work})()
    result = ActionResult(
        ok=True,
        summary="Bash script executed.",
        evidence={"stdout": "", "stderr": "", "exit_code": 0, "execution_type": "host"},
    )
    update = validate_node(
        {
            "current_step": 0,
            "max_retries": 0,
            "test_matrix": [step.model_dump()],
            "step_views": [StepView.from_step(step).model_dump()],
            "last_result": result.model_dump(),
            "execution_history": [],
            "attempt_counts": {},
        },
        deps,
    )
    assert update["phase"] == "route"
    assert judge.calls[0][0] == statement
    assert judge.calls[0][1] == 0
    assert judge.calls[0][4] is False
    assert judge.calls[1][4] is True
    assert "Running on http://0.0.0.0:8080" in judge.calls[1][2]
    assert judge.calls[1][1] == 0


def test_retry_then_assertion_fail(tmp_path: Path) -> None:
    gui = StaticGUI("nope")
    service = _service(tmp_path, gui=gui)
    spec = "GUI browser: Submit the form\nAssertion: contains:registered\n"
    finished = service.wait(service.submit(spec)["id"])
    assert gui.calls == 1
    assert finished["status"] == "completed"
    assert finished["report"]["verdict"] == "fail"
    assert finished["report"]["reason_code"] == "assertion_failed"
    assert finished["report"]["steps"][0]["assertion_passed"] is False


def test_a_failed_step_is_not_retried(tmp_path: Path) -> None:
    class Advisor(LabeledPlanner):
        def __init__(self) -> None:
            self.calls = 0

        def fix_step(self, step: dict, error: str, retry_count: int) -> dict:
            del error, retry_count
            self.calls += 1
            return dict(step)

    gui = StaticGUI("nope")
    advisor = Advisor()
    service = _service(tmp_path, gui=gui, planner=advisor)
    spec = "GUI browser: Submit the form\nAssertion: contains:registered\n"
    finished = service.wait(service.submit(spec)["id"])
    assert advisor.calls == 0
    assert gui.calls == 1
    assert finished["report"]["verdict"] == "fail"
    assert finished["report"]["reason_code"] == "assertion_failed"


def test_a_revised_step_updates_its_phase_and_keeps_the_others(tmp_path: Path) -> None:
    class Advisor(LabeledPlanner):
        def plan(self, specification: str):
            from aqe.state import TestPhase

            result = LabeledPlanner.plan(self, specification)
            result.phases = [
                TestPhase(
                    phase=1,
                    name="Submit the form",
                    interface="GUI",
                    gui_driver="browser",
                    operation_notes=["click Submit"],
                    verifications=["contains:registered"],
                ),
                TestPhase(
                    phase=2,
                    name="Confirm the inbox",
                    interface="CLI",
                    operation_notes=["echo inbox"],
                    verifications=["contains:inbox"],
                ),
            ]
            return result

        def fix_step(self, step: dict, error: str, retry_count: int) -> dict:
            del error, retry_count
            raise AssertionError("a failed step is not sent back to the model")

    gui = StaticGUI("nope")
    service = _service(tmp_path, gui=gui, planner=Advisor())
    spec = "GUI browser: Submit the form\nAssertion: contains:registered\n"
    finished = service.wait(service.submit(spec)["id"])
    stored = PlanStorageManager(service.config.runs_dir).load_plan(finished["id"])
    assert stored is not None
    assert stored.phases[1].name == "Confirm the inbox"
    assert stored.phases[1].operation_notes == ["echo inbox"]
    assert finished["report"]["verdict"] == "fail"
    assert gui.calls == 1


def test_a_hard_failure_is_not_retried(tmp_path: Path) -> None:
    class Advisor(LabeledPlanner):
        def __init__(self) -> None:
            self.calls = 0

        def fix_step(self, step: dict, error: str, retry_count: int) -> dict:
            del error, retry_count
            self.calls += 1
            stopped = dict(step)
            stopped["_should_not_retry"] = True
            stopped["_retry_judgment"] = "The port is already in use."
            return stopped

    gui = StaticGUI("nope")
    advisor = Advisor()
    service = _service(tmp_path, gui=gui, planner=advisor)
    spec = "GUI browser: Submit the form\nAssertion: contains:registered\n"
    finished = service.wait(service.submit(spec)["id"])
    assert advisor.calls == 0
    assert gui.calls == 1
    assert finished["report"]["verdict"] == "fail"


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
    assert gui.calls == 1
    assert finished["report"]["verdict"] == "fail"
    assert finished["report"]["reason"] == explanation
    assert finished["report"]["steps"][0]["judgment"] == explanation


def test_a_judgment_is_requested_again_until_it_parses() -> None:
    from langchain_core.messages import AIMessage

    from aqe.judge import ChatPageJudge

    class Reply:
        def __init__(self) -> None:
            self.calls = 0
            self.queries: list[str] = []

        def invoke(self, messages: list) -> AIMessage:
            self.calls += 1
            self.queries.append(messages[-1].content)
            return AIMessage(content="The page looks fine.")

    judge = ChatPageJudge(Reply())
    try:
        judge.judge("The page contains the expected text.", "<html>expected</html>")
    except ValueError as exc:
        assert "valid JSON" in str(exc) or "passed" in str(exc)
    else:
        raise AssertionError("an unreadable judgment is not requested again")
    assert judge.model.calls == 1


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
    shown = present_for_judge(command, "The curl command executes.")
    assert "stdout length: 20000 characters" in shown
    assert "stderr length: 4 characters" in shown
    assert "boom" in shown
    recorded = command + "\n\nexit_code: 0\n\nscript_result:\nVIRTUAL_ENV=/work/venv\nvenv_dir=present\n"
    about_venv = present_for_judge(recorded, "A virtual environment named venv has been created and activated.")
    assert "VIRTUAL_ENV=/work/venv" in about_venv
    assert "venv_dir=present" in about_venv
    assert "stdout length:" in about_venv
    assert "stderr length:" in about_venv
    assert "exit_code" not in about_venv
    with_code = "stdout:\nhello\n\nstderr:\n\n\nexit_code: 1"
    both = present_for_judge(with_code, "The command output contains hello.")
    assert "stdout length: 5 characters" in both
    assert "hello" in both
    assert "stderr length: 0 characters" in both
    assert "exit_code" not in both


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


def test_a_requested_coding_phase_is_kept() -> None:
    from aqe.llm import _missing_coding_phase, _validate_phases
    from aqe.state import CodingAction, TestPhase

    cli_only = [
        TestPhase(
            phase=1,
            name="Write a file",
            interface="CLI",
            operation_notes=["cat > hello.py"],
            verifications=["The file exists."],
        )
    ]
    finding = _missing_coding_phase("Make a CODING phase that writes hello.py", cli_only)
    assert finding is not None
    assert "CODING" in finding
    coding = [
        TestPhase(
            phase=1,
            name="Write a file",
            interface="CODING",
            verifications=["The file hello.py exists."],
            coding_operations=[
                CodingAction(action="create_file", file_path="hello.py", content="print('hi')\n")
            ],
        )
    ]
    assert _missing_coding_phase("Make a CODING phase that writes hello.py", coding) is None
    assert _validate_phases(coding) is None
    missing_ops = _validate_phases(
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
    assert missing_ops is not None
    assert "coding operations" in missing_ops
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
    assert "desktop application" in desktop.lower()


def test_coding_operations_accept_the_models_field_names() -> None:
    from aqe.llm import _coerce_phases

    phases, problem = _coerce_phases(
        [
            {
                "phase": 1,
                "name": "Write the API",
                "interface": "CODING",
                "coding_operations": [
                    {
                        "create_file": {
                            "file_path": "app.py",
                            "content": "print('ok')\n",
                            "description": "API with a /health endpoint.",
                        }
                    },
                    {
                        "review_code": {
                            "file_path": "app.py",
                            "description": "Check the health endpoint is correctly implemented.",
                        }
                    },
                ],
                "verifications": ["The file app.py contains a /health endpoint."],
            },
            {
                "phase": 2,
                "name": "Write the health check",
                "interface": "CODING",
                "depends_on": [1],
                "coding_operations": [
                    {
                        "coding_operation": "create_file",
                        "file_path": "health.py",
                        "content": "ok\n",
                        "description": "health check endpoint.",
                    }
                ],
                "verifications": ["The file health.py contains the health check endpoint."],
            },
            {
                "phase": 3,
                "name": "Write app.py",
                "interface": "CODING",
                "depends_on": [],
                "coding_operations": [
                    {
                        "file_path": "app.py",
                        "content": "from flask import Flask\napp = Flask(__name__)\n",
                        "description": "Create the Python source file app.py containing a Flask application with a /health endpoint running on port 8080.",
                    }
                ],
                "verifications": ["The file app.py defines a /health endpoint."],
            },
        ]
    )
    assert problem is None
    assert [op.action for op in phases[0].coding_operations] == ["create_file", "review_code"]
    assert phases[0].coding_operations[0].file_path == "app.py"
    assert phases[0].coding_operations[0].content == "print('ok')\n"
    assert phases[0].coding_operations[1].description == "Check the health endpoint is correctly implemented."
    assert phases[1].coding_operations[0].action == "create_file"
    assert phases[1].coding_operations[0].file_path == "health.py"
    assert phases[2].coding_operations[0].action == "create_file"
    assert phases[2].coding_operations[0].content.startswith("from flask import Flask")


def test_the_model_phase_list_is_the_plan() -> None:
    from aqe.llm import _apply_phase_update
    from aqe.state import TestPhase

    current = [
        TestPhase(
            phase=1,
            name="Register",
            interface="GUI",
            gui_driver="browser",
            operation_notes=["open the form"],
            verifications=["The page contains registered."],
        ),
        TestPhase(
            phase=2,
            name="Check the log",
            interface="CLI",
            operation_notes=["cat log"],
            verifications=["The log contains ok."],
        ),
        TestPhase(
            phase=3,
            name="List files",
            interface="CLI",
            operation_notes=["ls"],
            verifications=["The command prints the file name."],
        ),
    ]
    returned = [
        {
            "phase": 1,
            "name": "Register",
            "interface": "GUI",
            "gui_driver": "browser",
            "operation_notes": ["open the form"],
            "verifications": ["The page contains registered."],
        },
        {
            "phase": 2,
            "name": "Check the log",
            "interface": "CLI",
            "operation_notes": ["cat service.log"],
            "verifications": ["The log file contains the word ready."],
        },
    ]
    phases, _, problem = _apply_phase_update(current, returned, "Remove phase 3 and update phase 2.")
    assert problem is None
    assert [phase.name for phase in phases] == ["Register", "Check the log"]
    assert phases[1].operation_notes == ["cat service.log"]


def test_the_stored_plan_is_the_model_phase_list() -> None:
    from aqe.llm import _apply_phase_update
    from aqe.state import TestPhase

    current = [
        TestPhase(
            phase=1,
            name="Create API",
            interface="CODING",
            coding_operations=[
                {
                    "action": "create_file",
                    "file_path": "app.py",
                    "content": "print('ok')\n",
                    "description": "API",
                }
            ],
            verifications=["The file app.py exists."],
        ),
        TestPhase(
            phase=2,
            name="Create virtual environment",
            interface="CLI",
            depends_on=[1],
            script="python -m venv venv\n",
            verifications=["The virtual environment is created."],
        ),
        TestPhase(
            phase=3,
            name="Install dependencies",
            interface="CLI",
            depends_on=[2],
            script="pip install Flask\n",
            verifications=["Dependencies are installed."],
        ),
        TestPhase(
            phase=4,
            name="Start API Service",
            interface="CLI",
            depends_on=[3],
            script="python app.py &\n",
            verifications=["The API process is present."],
        ),
        TestPhase(
            phase=5,
            name="Verify Health Check Endpoint",
            interface="CLI",
            depends_on=[4],
            script="curl http://localhost:8080/health\n",
            verifications=["The health endpoint returns OK."],
        ),
    ]
    raw = [
        {
            "phase": 1,
            "name": "Create API",
            "interface": "CODING",
            "coding_operations": [
                {
                    "action": "create_file",
                    "file_path": "app.py",
                    "content": "print('ok')\n",
                    "description": "API",
                }
            ],
            "verifications": ["The file app.py exists."],
        },
        {
            "phase": 2,
            "name": "Setup Environment and Start API Service",
            "interface": "CLI",
            "depends_on": [1],
            "script": "python -m venv venv\nsource venv/bin/activate\npip install Flask\npython app.py &\n",
            "verifications": ["The API process is present."],
        },
        {
            "phase": 3,
            "name": "Verify Health Check Endpoint",
            "interface": "CLI",
            "depends_on": [2],
            "script": "curl http://localhost:8080/health\n",
            "verifications": ["The health endpoint returns OK."],
        },
    ]
    phases, _, problem = _apply_phase_update(
        current,
        raw,
        "phase 2, phase 3, and phase 4 need to merge together so they share the same execution environment.",
    )
    assert problem is None
    assert [phase.name for phase in phases] == [
        "Create API",
        "Setup Environment and Start API Service",
        "Verify Health Check Endpoint",
    ]
    broken, _, problem = _apply_phase_update(
        current,
        [phase.model_dump() for phase in current if phase.phase != 4],
        "remove phase 4 and Phase 5",
    )
    assert broken == []
    assert problem is not None
    assert "phase 4" in problem


def test_removing_several_phases_drops_each_named_phase() -> None:
    from aqe.llm import _apply_phase_update
    from aqe.state import TestPhase

    current = [
        TestPhase(
            phase=1,
            name="Create API",
            interface="CODING",
            coding_operations=[
                {
                    "action": "create_file",
                    "file_path": "app.py",
                    "content": "print('ok')\n",
                    "description": "API",
                }
            ],
            verifications=["The file app.py exists."],
        ),
        TestPhase(
            phase=2,
            name="Setup",
            interface="CLI",
            depends_on=[1],
            script="python -m venv venv\n",
            verifications=["The virtual environment is created."],
        ),
        TestPhase(
            phase=3,
            name="Check",
            interface="CLI",
            depends_on=[2],
            script="curl http://localhost:8080/health\n",
            verifications=["The health endpoint returns OK."],
        ),
        TestPhase(
            phase=4,
            name="Start API Service",
            interface="CLI",
            depends_on=[3],
            script="python app.py &\n",
            verifications=["The API process is present."],
        ),
        TestPhase(
            phase=5,
            name="Verify Health Check Endpoint",
            interface="CLI",
            depends_on=[4],
            script="curl http://localhost:8080/health\n",
            verifications=["The health endpoint returns OK."],
        ),
    ]
    phases, _, problem = _apply_phase_update(
        current,
        [phase.model_dump() for phase in current if phase.phase <= 3],
        "remove phase 4 and Phase 5",
    )
    assert problem is None
    assert [phase.phase for phase in phases] == [1, 2, 3]


def test_coding_phase_is_judged_from_the_written_file(tmp_path: Path) -> None:
    from aqe.state import CodingAction, TestStep

    question = "The file hello.py contains print('hello')."

    class CodingPlanner:
        def __init__(self, content: str) -> None:
            self.content = content

        def plan(self, specification: str):
            del specification
            return PlanResult(
                accepted=True,
                steps=[
                    TestStep(
                        step=1,
                        interface="CODING",
                        action="Create hello.py",
                        assertion=question,
                        verifications=[question],
                        coding_operations=[
                            CodingAction(
                                action="create_file",
                                file_path="hello.py",
                                content=self.content,
                                description="Write hello.py",
                            )
                        ],
                    )
                ],
            )

        def script_for(self, intent: str) -> str:
            del intent
            return "print('ok')\n"

    class FileJudge:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def judge(self, asked: str, text: str) -> Judgment:
            self.calls.append((asked, text))
            passed = "print('hello')" in text and "file: hello.py" in text
            detail = (
                "The file hello.py contains print('hello')."
                if passed
                else "The file hello.py does not contain print('hello')."
            )
            return Judgment(passed=passed, judgment=detail)

    matched = FileJudge()
    service = _service(tmp_path, planner=CodingPlanner("print('hello')\n"), judge=matched)
    finished = service.wait(service.submit("Create hello.py")["id"])
    assert finished["report"]["verdict"] == "pass"
    assert matched.calls
    asked, text = matched.calls[0]
    assert asked == question
    assert "file: hello.py" in text
    assert "content:\nprint('hello')" in text
    assert "Executed" not in text

    missed = FileJudge()
    other = _service(tmp_path, planner=CodingPlanner("print('bye')\n"), judge=missed)
    failed = other.wait(other.submit("Create hello.py")["id"])
    assert failed["report"]["verdict"] == "fail"
    assert missed.calls
    assert "print('bye')" in missed.calls[0][1]
    assert failed["report"]["steps"][0]["assertion_passed"] is False
