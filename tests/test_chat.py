"""Chat model selection from the environment."""

import pytest

from aqe.chat import get_chat_model
from aqe.llm import NONSENSE_SPEC
from aqe.settings import get_settings

SAMPLE = """# User registration webhook

Verify that registration through the GUI records a webhook.
"""


def _reset(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    get_chat_model.cache_clear()


def test_openai_without_credentials_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset(
        monkeypatch,
        LLM_PROVIDER="openai",
        LLM_MODEL="gpt-4o-mini",
        OPENAI_API_KEY="",
        AIA_GATEWAY_CLIENT_ID="",
        AIA_GATEWAY_CLIENT_SECRET="",
        AIA_GATEWAY_BASE_URL="",
        REALLM_BASE_URL="",
        REALLM_API_KEY="",
    )
    with pytest.raises(ValueError, match="No valid OpenAI configuration"):
        get_chat_model()


def test_run_command_reads_spec_path(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _reset(
        monkeypatch,
        LLM_PROVIDER="ollama",
        LLM_MODEL="gemma4:e2b",
        SPEC_PATH="examples/specs/registration.md",
        RUNS_DIR=str(tmp_path / "runs"),
    )

    class FakeService:
        def __init__(self, config) -> None:
            self.config = config

        def run_dir(self, run_id: str):
            path = tmp_path / "runs" / run_id
            path.mkdir(parents=True, exist_ok=True)
            return path

        def submit(self, specification: str, run_id: str | None = None) -> dict:
            del specification, run_id
            return {}

        def wait(self, run_id: str) -> dict:
            report_path = self.run_dir(run_id) / "report.json"
            report_path.write_text("{}", encoding="utf-8")
            return {"report": {"verdict": "pass"}}

    monkeypatch.setattr("aqe.__main__.RunService", FakeService)
    monkeypatch.setattr("aqe.__main__._start_sut", lambda *_args: None)
    from aqe.__main__ import main

    assert main(["run"]) == 0
    assert list((tmp_path / "runs").glob("*/report.json"))


def test_run_command_requires_spec_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset(monkeypatch, LLM_PROVIDER="ollama", LLM_MODEL="gemma4:e2b", SPEC_PATH="")
    from aqe.__main__ import main

    assert main(["run"]) == 2


def test_plan_without_accepted_is_accepted_when_steps_exist() -> None:
    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner

    class Reply:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages: list) -> AIMessage:
            del messages
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="""```json
{
  "reason": "GUI then CLI",
  "steps": [
    {
      "step": 1,
      "interface": "GUI",
      "gui_driver": "browser",
      "action": "Open the registration form, enter user ada, and submit.",
      "assertion": "contains:registered"
    },
    {
      "step": 2,
      "interface": "CLI",
      "gui_driver": null,
      "action": "Confirm the webhook file contains that user.",
      "assertion": "json:user=ada"
    }
  ]
}
```"""
                )
            return AIMessage(
                content=(
                    '{"operations": [{"step": 1, "operations": ['
                    '{"action": "goto", "text": "http://127.0.0.1:8765"}, '
                    '{"action": "type", "selector": {"role": "textbox", "name": "Username"}, "text": "ada"}, '
                    '{"action": "click", "selector": {"role": "button", "name": "Register"}}]}]}'
                )
            )

    result = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").plan(SAMPLE)
    assert result.accepted
    assert [step.interface for step in result.steps] == ["GUI", "CLI"]
    assert [op.action for op in result.steps[0].operations] == ["goto", "type", "click"]


def test_plan_coerces_string_steps_and_action_objects() -> None:
    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner

    class Reply:
        def invoke(self, messages: list) -> AIMessage:
            del messages
            return AIMessage(
                content=(
                    '{"accepted": true, "reason": null, "steps": ['
                    '{"step": "Open the form", "interface": "GUI", "gui_driver": "browser", '
                    '"action": {"goto": "http://127.0.0.1:8765", "type": "text", '
                    '"selector": {"role": "textbox", "name": "Username"}, "text": "ada"}, '
                    '"assertion": "contains:registered"}, '
                    '{"step": "Check the webhook", "interface": "CLI", "gui_driver": null, '
                    '"action": "read the webhook", "assertion": "json:user=ada"}]}'
                )
            )

    result = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").plan(SAMPLE)
    assert result.accepted
    gui = result.steps[0]
    assert gui.step == 1
    assert gui.action == "Open the form"
    assert [op.action for op in gui.operations] == ["goto", "type"]


def test_split_gui_visit_becomes_one_step() -> None:
    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner

    class Reply:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages: list) -> AIMessage:
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content=(
                        '{"accepted": true, "steps": ['
                        '{"step": 1, "interface": "GUI", "gui_driver": "browser", "action": "navigate_to", "assertion": "contains:http://localhost:8765"}, '
                        '{"step": 2, "interface": "GUI", "gui_driver": "browser", "action": "enter_text", "assertion": "json:username=ada"}, '
                        '{"step": 3, "interface": "GUI", "gui_driver": "browser", "action": "submit_form", "assertion": "status:200"}, '
                        '{"step": 4, "interface": "GUI", "gui_driver": "browser", "action": "assert_text_content", "assertion": "contains:registered"}'
                        "]}"
                    )
                )
            assert "Step 2" not in messages[-1].content
            return AIMessage(
                content=(
                    '{"operations": [{"step": 1, "operations": ['
                    '{"action": "goto", "text": "http://localhost:8765"}, '
                    '{"action": "type", "selector": {"role": "textbox", "name": "Username"}, "text": "ada"}, '
                    '{"action": "click", "selector": {"role": "button", "name": "Register"}}]}]}'
                )
            )

    result = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").plan("register ada")
    assert result.accepted
    assert len(result.steps) == 1
    assert result.steps[0].assertion == "contains:registered"
    assert [op.action for op in result.steps[0].operations] == ["goto", "type", "click"]


def test_plan_accepts_trailing_commas_and_single_quotes() -> None:
    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner

    class Reply:
        def invoke(self, messages: list) -> AIMessage:
            del messages
            return AIMessage(
                content=(
                    "{'accepted': true, 'reason': null, 'steps': ["
                    "{'step': 1, 'interface': 'GUI', 'gui_driver': 'browser', "
                    "'action': 'Open the form', 'assertion': 'contains:registered', "
                    "'operations': [{'action': 'click', 'selector': {'role': 'button', 'name': 'Register'},},]},"
                    "]}"
                )
            )

    result = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").plan("Open the form")
    assert result.accepted
    assert result.steps[0].operations[0].action == "click"


def test_verbose_verification_keeps_the_original_sentence() -> None:
    from aqe.llm import verbalize_verification

    original = (
        "The page shows the word registered after the form is submitted, "
        "and it does not show an error message."
    )
    assert verbalize_verification("submit the form and look for registered", original) == original
    assert verbalize_verification(
        "the page should contain registered",
        "contains:registered",
    ) == 'The result contains the text "registered". No other substitution satisfies this check.'


def test_cli_verifications_must_name_the_command_streams() -> None:
    from aqe.llm import _PLAN_SYSTEM, _REFINE_PLAN_SYSTEM, _REPAIR_SYSTEM

    for prompt in (_PLAN_SYSTEM, _REFINE_PLAN_SYSTEM, _REPAIR_SYSTEM):
        assert "$? " in prompt
        assert "vague" in prompt
        assert "background service is running" in prompt


def test_stdout_claim_is_removed_when_the_request_does_not_mention_output() -> None:
    from aqe.llm import omit_unrequested_stdout
    from aqe.state import TestPhase

    phase = TestPhase(
        phase=1,
        name="curl",
        interface="CLI",
        verifications=[
            "The curl command executes and returns its output. stdout is not empty. There must be an error."
        ],
        operation_notes=["curl http://example"],
    )
    cleaned = omit_unrequested_stdout("Verify the curl command reports an error.", [phase])
    text = cleaned[0].verifications[0]
    assert "stdout" not in text.lower()
    assert "output" not in text.lower()
    assert "executes" in text.lower()
    assert "error" in text.lower()
    kept = omit_unrequested_stdout("Verify that stdout is not empty.", [phase])
    assert "stdout is not empty" in kept[0].verifications[0]


def test_returns_nothing_stays_an_empty_output_check() -> None:
    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner

    class Reply:
        def invoke(self, messages: list) -> AIMessage:
            del messages
            return AIMessage(
                content=(
                    '{"accepted": true, "phases": [{"phase": 1, "name": "Run ls command", '
                    '"interface": "CLI", "operations": ["ls"], "verifications": ["status:CODE=0"]}]}'
                )
            )

    result = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").plan(
        "Run ls command. Verify that it returns nothing."
    )
    assert result.accepted
    assert result.steps[0].verifications == [
        (
            "The command returns nothing. stdout is empty. "
            "A successful exit code does not satisfy this check."
        )
    ]
    assert result.steps[0].action == "ls"


def test_run_ls_command_executes_ls() -> None:
    from aqe.llm import ChatModelPlanner

    class Reply:
        def invoke(self, messages: list):
            del messages
            raise AssertionError("run ls command does not ask the model for a script")

    script = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").script_for(
        "run ls command\nAssertion: The standard output of the command must be empty."
    )
    assert "subprocess.run" in script
    assert "'ls'" in script
    assert "'run'" not in script


def test_whoami_is_a_command_not_python() -> None:
    from aqe.llm import ChatModelPlanner

    class Reply:
        def invoke(self, messages: list):
            del messages
            raise AssertionError("whoami does not ask the model for a script")

    script = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").script_for(
        "whoami\nAssertion: The output contains a username"
    )
    assert "subprocess.run" in script
    assert "'whoami'" in script
    assert "NameError" not in script


def test_curl_command_is_the_script() -> None:
    from aqe.llm import ChatModelPlanner, command_needs_network

    class Reply:
        def invoke(self, messages: list):
            del messages
            raise AssertionError("a curl operation does not ask the model for a script")

    script = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").script_for(
        "curl google.com\nAssertion: The output contains text"
    )
    assert "webhook.json" not in script
    assert "curl" in script
    assert "google.com" in script
    assert command_needs_network("curl google.com") is True
    assert command_needs_network("echo hello") is False


def test_labeled_notes_run_the_shell_commands() -> None:
    from aqe.llm import ChatModelPlanner

    class Reply:
        def invoke(self, messages: list):
            del messages
            raise AssertionError("labeled shell notes do not ask the model for a script")

    notes = (
        "Create and activate virtual environment: python -m venv venv && source venv/bin/activate. "
        "Install dependencies: pip install -r requirements.txt\n"
        "Assertion: Dependencies are installed successfully within the virtual environment."
    )
    script = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").script_for(notes)
    assert "shell=True" in script
    assert "python -m venv venv && source venv/bin/activate && pip install -r requirements.txt" in script
    assert "Create" not in script
    assert "Install" not in script


def test_json_assertion_reads_evidence_file() -> None:
    from aqe.llm import ChatModelPlanner

    class Reply:
        def invoke(self, messages: list):
            del messages
            raise AssertionError("json assertions do not ask the model for a script")

    script = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").script_for(
        "Confirm the webhook file contains that user.\nAssertion: json:user=ada"
    )
    assert "/evidence" in script
    assert "print" in script


def test_phases_run_in_dependency_order() -> None:
    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner

    class Reply:
        def invoke(self, messages: list) -> AIMessage:
            del messages
            return AIMessage(
                content=(
                    '{"accepted": true, "phases": ['
                    '{"phase": 2, "name": "Confirm the webhook", "depends_on": [1], "interface": "CLI", '
                    '"operations": ["Read the webhook file"], "verifications": ["json:user=ada"]}, '
                    '{"phase": 1, "name": "Register ada", "depends_on": [], "interface": "GUI", '
                    '"gui_driver": "browser", "operations": ['
                    '{"action": "goto", "text": "http://127.0.0.1:8765"}, '
                    '{"action": "type", "selector": {"role": "textbox", "name": "Username"}, "text": "ada"}, '
                    '{"action": "click", "selector": {"role": "button", "name": "Register"}}'
                    '], "verifications": ["The page contains only the word registered", '
                    '"The page does not show an error"]}]}'
                )
            )

    result = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").plan("register then check the webhook")
    assert result.accepted
    assert [step.phase_name for step in result.steps] == ["Register ada", "Confirm the webhook"]
    assert result.steps[1].depends_on == [1]
    assert result.steps[0].verifications == [
        "The page contains only the word registered",
        "The page does not show an error",
    ]
    assert [op.action for op in result.steps[0].operations] == ["goto", "type", "click"]
    assert result.steps[1].operation_notes == ["Read the webhook file"]


def test_incomplete_phase_plan_is_repaired_into_dependency_order() -> None:
    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner

    draft = """{
      "accepted": true,
      "reason": "The webhook check is assumed to be implicit because no interface was provided.",
      "phases": [{
        "phase": 1,
        "name": "Register ada",
        "interface": "GUI",
        "gui_driver": "browser",
        "operations": [{"action": "goto", "text": "http://127.0.0.1:8765"}],
        "verifications": [{"type": "page_check", "description": "The page contains only the word registered"}]
      }]
    }"""
    revised = (
        '{"accepted": true, "phases": ['
        '{"phase": 2, "name": "Confirm the webhook", "depends_on": [1], "interface": "CLI", '
        '"operations": ["Read the webhook file"], "verifications": ["json:user=ada"]}, '
        '{"phase": 1, "name": "Register ada", "depends_on": [], "interface": "GUI", "gui_driver": "browser", '
        '"operations": [{"action": "goto", "text": "http://127.0.0.1:8765"}], '
        '"verifications": ["The page contains only the word registered"]}]}'
    )

    class Reply:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages: list) -> AIMessage:
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content=draft)
            assert "implicit" in messages[-1].content
            return AIMessage(content=revised)

    result = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").plan("register ada and confirm the webhook")
    assert result.accepted
    assert [step.phase_name for step in result.steps] == ["Register ada", "Confirm the webhook"]
    assert result.steps[1].depends_on == [1]
    assert result.steps[0].verifications == ["The page contains only the word registered"]


def test_circular_phases_are_rejected_with_an_explanation() -> None:
    from aqe.llm import order_phases
    from aqe.state import TestPhase

    phases = [
        TestPhase(phase=1, name="Create the user", interface="GUI", gui_driver="browser", depends_on=[2], verifications=["created"], operation_notes=["submit"]),
        TestPhase(phase=2, name="Read the user", interface="CLI", depends_on=[1], verifications=["json:user=ada"], operation_notes=["read"]),
    ]
    ordered, problem = order_phases(phases)
    assert ordered == []
    assert problem is not None
    assert "circle" in problem
    assert "Create the user" in problem
    assert "Read the user" in problem


def test_cli_phase_without_verifications_uses_its_operation() -> None:
    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner

    class Reply:
        def invoke(self, messages: list) -> AIMessage:
            del messages
            return AIMessage(
                content=(
                    '{"accepted": true, "phases": ['
                    '{"phase": 1, "name": "Register ada", "interface": "GUI", "gui_driver": "browser", '
                    '"operations": [{"action": "goto", "text": "http://127.0.0.1:8765"}], '
                    '"verifications": ["The page contains only the word registered"]}, '
                    '{"phase": 2, "name": "Confirm the webhook", "depends_on": [1], "interface": "CLI", '
                    '"operations": ["check_webhook_file_for_user(\'ada\')"], "verifications": []}]}'
                )
            )

    result = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").plan("register ada then check the webhook")
    assert result.accepted
    assert result.steps[1].depends_on == [1]
    assert result.steps[1].verifications == ["check_webhook_file_for_user('ada')"]
    assert result.steps[1].operation_notes == ["check_webhook_file_for_user('ada')"]


def test_cli_phase_without_operations_uses_its_verification() -> None:
    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner

    class Reply:
        def invoke(self, messages: list) -> AIMessage:
            del messages
            return AIMessage(
                content=(
                    '{"accepted": true, "phases": ['
                    '{"phase": 1, "name": "Register ada", "interface": "GUI", "gui_driver": "browser", '
                    '"operations": [{"action": "goto", "text": "http://127.0.0.1:8765"}], '
                    '"verifications": ["The page contains only the word registered"]}, '
                    '{"phase": 2, "name": "Confirm the webhook", "depends_on": [1], "interface": "CLI", '
                    '"operations": [], "verifications": ["json:user=ada"]}]}'
                )
            )

    result = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").plan("register ada then check the webhook")
    assert result.accepted
    assert result.steps[1].phase_name == "Confirm the webhook"
    assert result.steps[1].depends_on == [1]
    assert result.steps[1].operation_notes == ["Carry out: json:user=ada"]
    assert "user" in result.steps[1].verifications[0]
    assert "ada" in result.steps[1].verifications[0]
    assert result.steps[1].verifications[0].startswith("The JSON output")


def test_phase_without_verifications_explains_the_gap() -> None:
    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner

    class Reply:
        def invoke(self, messages: list) -> AIMessage:
            del messages
            return AIMessage(
                content=(
                    '{"accepted": true, "phases": [{"phase": 1, "name": "Register ada", '
                    '"interface": "GUI", "gui_driver": "browser", '
                    '"operations": ["Open the form and submit ada"], "verifications": []}]}'
                )
            )

    result = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").plan("register ada")
    assert result.accepted is False
    assert result.reason_code == "not_a_test_plan"
    assert result.reason is not None
    assert "no verifications" in result.reason
    assert "Register ada" in result.reason


def test_refusal_to_execute_is_asked_to_plan_again() -> None:
    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner

    class Reply:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages: list) -> AIMessage:
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content=(
                        '{"accepted": false, "phases": [], "reason": '
                        '"As an LLM, I do not have the capability to execute live browser automation."}'
                    )
                )
            assert "planner, not the executor" in messages[-1].content
            return AIMessage(
                content=(
                    '{"accepted": true, "phases": ['
                    '{"phase": 1, "name": "Register ada", "interface": "GUI", "gui_driver": "browser", '
                    '"operations": [{"action": "goto", "text": "http://127.0.0.1:8765"}], '
                    '"verifications": ["The page contains only the word registered"]}, '
                    '{"phase": 2, "name": "Confirm the webhook", "depends_on": [1], "interface": "CLI", '
                    '"operations": ["Read the webhook file"], "verifications": ["json:user=ada"]}]}'
                )
            )

    result = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").plan("register ada then check the webhook")
    assert result.accepted
    assert [step.phase_name for step in result.steps] == ["Register ada", "Confirm the webhook"]
    assert result.steps[1].depends_on == [1]


def test_check_without_operations_joins_the_action_phase() -> None:
    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner

    class Reply:
        def invoke(self, messages: list) -> AIMessage:
            del messages
            return AIMessage(
                content=(
                    '{"accepted": true, "phases": ['
                    '{"phase": 1, "name": "Register ada", "interface": "GUI", "gui_driver": "browser", '
                    '"operations": ['
                    '{"action": "goto", "text": "http://localhost:8765"}, '
                    '{"action": "type", "selector": {"role": "textbox", "name": "Username"}, "text": "ada"}, '
                    '{"action": "click", "selector": {"role": "button", "name": "Commit"}}], '
                    '"verifications": []}, '
                    '{"phase": 2, "name": "Check the page", "depends_on": [1], "interface": "GUI", '
                    '"gui_driver": "browser", "operations": [], '
                    '"verifications": ["The page contains registered"]}]}'
                )
            )

    result = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").plan(
        'Navigate to http://localhost:8765, enter "ada" in Username, click Commit, and verify the page contains registered.'
    )
    assert result.accepted
    assert len(result.steps) == 1
    assert result.steps[0].verifications == ["The page contains registered"]
    assert [op.action for op in result.steps[0].operations] == ["goto", "type", "click"]
    assert result.steps[0].operations[1].selector == {"role": "textbox", "name": "Username"}
    assert result.steps[0].operations[2].selector == {"role": "button", "name": "Commit"}


def test_structural_rejection_is_replanned() -> None:
    import json

    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner

    essay = (
        "The request asked to open a page, type ada, click Commit, and verify registered. "
        "The verification step cannot logically exist as a standalone phase."
    )

    class Reply:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages: list) -> AIMessage:
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content=json.dumps({"accepted": False, "reason": essay, "phases": []}))
            assert "Do not reject that sequence" in messages[-1].content
            return AIMessage(
                content=(
                    '{"accepted": true, "phases": [{"phase": 1, "name": "Register ada", "interface": "GUI", '
                    '"gui_driver": "browser", "operations": ['
                    '{"action": "goto", "text": "http://localhost:8765"}, '
                    '{"action": "type", "selector": {"role": "textbox", "name": "Username"}, "text": "ada"}, '
                    '{"action": "click", "selector": {"role": "button", "name": "Commit"}}], '
                    '"verifications": ["The page contains registered"]}]}'
                )
            )

    result = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").plan("click Commit and verify registered")
    assert result.accepted
    assert result.steps[0].verifications == ["The page contains registered"]
    assert result.steps[0].operations[2].selector["name"] == "Commit"


def test_running_api_refusal_is_replanned() -> None:
    import json

    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner

    reason = (
        "The request requires creating and verifying a running API service. "
        "While I can use the Pi coding agent to generate the source code files, "
        "I do not have the capability to execute this code to start a server or "
        "interact with a live network endpoint to perform the health check verification. "
        "Therefore, the verification phase cannot be executed using the available tools."
    )
    plan = {
        "accepted": True,
        "reason": None,
        "phases": [
            {
                "phase": 1,
                "name": "Write the API",
                "depends_on": [],
                "interface": "CODING",
                "coding_operations": [
                    {
                        "action": "create_file",
                        "file_path": "app.py",
                        "content": "print('ok')\n",
                        "description": "API service",
                    }
                ],
                "verifications": ["The file app.py contains the API service."],
            },
            {
                "phase": 2,
                "name": "Check the health endpoint",
                "depends_on": [1],
                "interface": "CLI",
                "operations": [
                    "nohup python app.py > service.log 2>&1 &",
                    "curl http://127.0.0.1:8080/health",
                ],
                "verifications": ["The health endpoint returns a successful response."],
            },
        ],
    }

    class Reply:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages: list) -> AIMessage:
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content=json.dumps({"accepted": False, "reason": reason, "phases": []}))
            assert "CODING phase" in messages[-1].content
            assert "health check" in messages[-1].content
            if self.calls == 2:
                return AIMessage(content=json.dumps({"accepted": False, "reason": reason, "phases": []}))
            assert "Set accepted to true" in messages[-1].content
            return AIMessage(content=json.dumps(plan))

    result = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").plan(
        "Create an API service and verify that its health endpoint is running."
    )
    assert result.accepted
    assert [step.interface for step in result.steps] == ["CODING", "CLI"]
    assert result.steps[1].depends_on == [1]
    assert "curl http://127.0.0.1:8080/health" in result.steps[1].operation_notes


def test_short_rejection_is_expanded() -> None:
    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner

    explanation = (
        "The request asks to confirm a webhook that registration never writes. "
        "Phase order cannot be built because the check has no producer: nothing in the request "
        "creates the file the later check reads, so the pipeline has a missing dependency."
    )

    class Reply:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages: list) -> AIMessage:
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content='{"accepted": false, "reason": "circular", "phases": []}')
            assert "circular" in messages[-1].content
            return AIMessage(content=explanation)

    result = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").plan("check a webhook that is never created")
    assert result.accepted is False
    assert result.reason == explanation


def test_unparsed_planner_output_is_rejected() -> None:
    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner

    class Reply:
        def invoke(self, messages: list) -> AIMessage:
            del messages
            return AIMessage(content=NONSENSE_SPEC)

    result = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").plan(SAMPLE)
    assert result.accepted is False
    assert result.reason_code == "not_a_test_plan"


def test_plan_chat_reads_json_with_a_missing_comma_and_a_raw_newline() -> None:
    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner
    from aqe.state import CodingAction, TestPhase, TestStep

    broken = """{
      "type": "plan_update",
      "phases": [
        {
          "phase": 1,
          "name": "Write the API",
          "interface": "CODING",
          "coding_operations": [
            {
              "action": "create_file",
              "file_path": "app.py",
              "content": "from flask import Flask
app = Flask(__name__)",
              "description": "Create the API"
            }
          ],
          "verifications": ["The file app.py contains Flask."]
        }
        {
          "phase": 2,
          "name": "Check health",
          "depends_on": [1],
          "interface": "CLI",
          "operations": ["curl http://127.0.0.1:8080/health"],
          "verifications": ["The health endpoint returns OK."]
        }
      ],
      "reasoning": "Updated the service file."
    }"""

    class Reply:
        def invoke(self, messages: list) -> AIMessage:
            del messages
            return AIMessage(content=broken)

    current = [
        TestPhase(
            phase=1,
            name="Write the API",
            interface="CODING",
            coding_operations=[
                CodingAction(action="create_file", file_path="app.py", content="print('ok')\n", description="API")
            ],
            verifications=["The file app.py contains Flask."],
        )
    ]
    steps = [
        TestStep(
            step=1,
            interface="CODING",
            action="Create app.py",
            assertion="The file app.py contains Flask.",
            verifications=["The file app.py contains Flask."],
            coding_operations=current[0].coding_operations,
        )
    ]
    phases, _, reasoning = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").refine_plan(
        current, steps, "Put Flask in app.py", []
    )
    assert "Failed to refine" not in reasoning
    assert [phase.phase for phase in phases] == [1, 2]
    assert "from flask import Flask\napp = Flask(__name__)" == phases[0].coding_operations[0].content
    assert phases[1].interface == "CLI"


def test_plan_chat_reads_source_that_contains_quotes() -> None:
    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner
    from aqe.state import CodingAction, TestPhase, TestStep

    source = """from flask import Flask

app = Flask(__name__)

@app.route("/health")
def health_check():
    return {"status": "ok"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
"""
    broken = (
        "{\n"
        '  "type": "plan_update",\n'
        '  "phases": [\n'
        "    {\n"
        '      "phase": 1,\n'
        '      "name": "Create API Service",\n'
        '      "interface": "CODING",\n'
        '      "coding_operations": [{\n'
        '        "action": "create_file",\n'
        '        "file_path": "app.py",\n'
        '        "content": "' + source + '",\n'
        '        "description": "Generate the Flask application."\n'
        "      }],\n"
        '      "verifications": ["The file app.py contains a /health endpoint."]\n'
        "    }\n"
        "    {\n"
        '      "phase": 2,\n'
        '      "name": "Install dependencies",\n'
        '      "interface": "CLI",\n'
        '      "operations": ["pip install flask"],\n'
        '      "verifications": ["Flask is installed."]\n'
        "    }\n"
        "  ],\n"
        '  "reasoning": "Install dependencies before the health check."\n'
        "}\n"
    )

    class Reply:
        def invoke(self, messages: list) -> AIMessage:
            del messages
            return AIMessage(content=broken)

    current = [
        TestPhase(
            phase=1,
            name="Create API Service",
            interface="CODING",
            coding_operations=[
                CodingAction(
                    action="create_file",
                    file_path="app.py",
                    content="print('ok')\n",
                    description="API",
                )
            ],
            verifications=["The file app.py contains a /health endpoint."],
        )
    ]
    steps = [
        TestStep(
            step=1,
            interface="CODING",
            action="Create API Service",
            assertion="The file app.py contains a /health endpoint.",
            verifications=["The file app.py contains a /health endpoint."],
            coding_operations=current[0].coding_operations,
        )
    ]
    phases, _, reasoning = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").refine_plan(
        current, steps, "Before phase 2, you should install dependencies for the API app.", []
    )
    assert "Failed to refine" not in reasoning
    assert [phase.name for phase in phases] == ["Create API Service", "Install dependencies"]
    assert '@app.route("/health")' in phases[0].coding_operations[0].content
    assert '{"status": "ok"}' in phases[0].coding_operations[0].content
    assert phases[1].operation_notes == ["pip install flask"]


def test_plan_update_accepts_a_list_closed_with_a_brace() -> None:
    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner
    from aqe.state import CodingAction, TestPhase, TestStep

    broken = """{
      "type": "plan_update",
      "phases": [
        {
          "phase": 1,
          "name": "Create API Service",
          "interface": "CODING",
          "depends_on": [],
          "operations": [
            {
              "action": "create_file",
              "file_path": "app.py",
              "content": "print('ok')\\n",
              "description": "Create the Flask application file app.py with a /health endpoint."
            }
          ],
          "verifications": ["The file app.py contains a /health endpoint."]
        },
        {
          "phase": 2,
          "name": "Install Dependencies",
          "interface": "CLI",
          "depends_on": [1],
          "operations": ["pip install -r requirements.txt"],
          "verifications": ["Dependencies for the API application have been installed successfully."]
        }
      ],
      "steps": [
        {
          "step": 2,
          "interface": "CLI",
          "action": "Install Dependencies",
          "assertion": "Dependencies are installed.",
          "verifications": ["Dependencies for the API application have been installed successfully."],
          "operations": ["pip install -r requirements.txt"]
        }
      },
      "reasoning": "Added Phase 2 to install dependencies before starting the service."
    }"""
    lines = broken.splitlines()
    for index, line in enumerate(lines):
        if '"reasoning"' in line:
            lines[index - 1] = "      },"
            break
    broken = "\n".join(lines)

    class Reply:
        def invoke(self, messages: list) -> AIMessage:
            del messages
            return AIMessage(content=broken)

    current = [
        TestPhase(
            phase=1,
            name="Create API Service",
            interface="CODING",
            coding_operations=[
                CodingAction(action="create_file", file_path="app.py", content="print('ok')\n", description="API")
            ],
            verifications=["The file app.py contains a /health endpoint."],
        ),
        TestPhase(
            phase=2,
            name="Start API Service",
            interface="CLI",
            depends_on=[1],
            operation_notes=["python app.py &"],
            verifications=["A python process running app.py is present."],
        ),
    ]
    steps = [
        TestStep(
            step=1,
            interface="CODING",
            action="Create API Service",
            assertion="The file app.py contains a /health endpoint.",
            verifications=["The file app.py contains a /health endpoint."],
            coding_operations=current[0].coding_operations,
        ),
        TestStep(
            step=2,
            interface="CLI",
            action="python app.py &",
            assertion="A python process running app.py is present.",
            verifications=["A python process running app.py is present."],
            operation_notes=["python app.py &"],
        ),
    ]
    phases, _, reasoning = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").refine_plan(
        current, steps, "Before phase 2, you should install dependencies for the API app.", []
    )
    assert "Failed to refine" not in reasoning
    assert [phase.name for phase in phases] == ["Create API Service", "Install Dependencies"]
    assert phases[1].operation_notes == ["pip install -r requirements.txt"]
    assert phases[0].coding_operations[0].file_path == "app.py"


def test_json_plan_update_inserts_a_phase() -> None:
    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner
    from aqe.state import CodingAction, TestPhase, TestStep

    class Reply:
        def invoke(self, messages: list) -> AIMessage:
            assert "one valid JSON object" in messages[0].content
            assert "Do not return JSON" not in messages[0].content
            return AIMessage(
                content=(
                    '{"type": "plan_update", "reasoning": "Added an install phase.", "phases": ['
                    '{"phase": 1, "name": "Create API Service", "interface": "CODING", '
                    '"coding_operations": [{"action": "create_file", "file_path": "app.py", '
                    '"content": "from flask import Flask\\n", "description": "Flask application"}], '
                    '"verifications": ["The file app.py contains a /health endpoint."]}, '
                    '{"phase": 2, "name": "Install dependencies", "interface": "CLI", "depends_on": [1], '
                    '"operation_notes": ["pip install flask"], "verifications": ["Flask is installed."]}, '
                    '{"phase": 3, "name": "Start API Service", "interface": "CLI", "depends_on": [2], '
                    '"operation_notes": ["python app.py &"], '
                    '"verifications": ["A python process running app.py is present."]}]}'
                )
            )

    current = [
        TestPhase(
            phase=1,
            name="Create API Service",
            interface="CODING",
            coding_operations=[
                CodingAction(
                    action="create_file",
                    file_path="app.py",
                    content="from flask import Flask\n",
                    description="Flask application",
                )
            ],
            verifications=["The file app.py contains a /health endpoint."],
        ),
        TestPhase(
            phase=2,
            name="Start API Service",
            interface="CLI",
            depends_on=[1],
            operation_notes=["python app.py &"],
            verifications=["A python process running app.py is present."],
        ),
    ]
    steps = [
        TestStep(
            step=1,
            interface="CODING",
            action="Create API Service",
            assertion="The file app.py contains a /health endpoint.",
            verifications=["The file app.py contains a /health endpoint."],
            coding_operations=current[0].coding_operations,
        ),
        TestStep(
            step=2,
            interface="CLI",
            action="python app.py &",
            assertion="A python process running app.py is present.",
            verifications=["A python process running app.py is present."],
            operation_notes=["python app.py &"],
        ),
    ]
    phases, _, reasoning = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").refine_plan(
        current, steps, "Before phase 2, you should install dependencies for the API app.", []
    )
    assert "Failed to refine" not in reasoning
    assert [phase.name for phase in phases] == [
        "Create API Service",
        "Install dependencies",
        "Start API Service",
    ]
    assert phases[1].interface == "CLI"
    assert phases[1].operation_notes == ["pip install flask"]
    assert phases[0].coding_operations[0].content.startswith("from flask import Flask")
    assert phases[2].operation_notes == ["python app.py &"]


def test_json_plan_update_is_applied_directly() -> None:
    import json

    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner
    from aqe.state import CodingAction, TestPhase, TestStep

    flask = "from flask import Flask\napp = Flask(__name__)\n"
    payload = {
        "type": "plan_update",
        "reasoning": "Create the virtual environment before installing Flask.",
        "phases": [
            {
                "phase": 1,
                "name": "Create API service code",
                "interface": "CODING",
                "coding_operations": [
                    {
                        "action": "create_file",
                        "file_path": "app.py",
                        "content": flask,
                        "description": "Flask application",
                    }
                ],
                "verifications": ["The file app.py contains a /health endpoint."],
            },
            {
                "phase": 2,
                "name": "Create virtual environment",
                "interface": "CLI",
                "depends_on": [1],
                "operation_notes": ["python3 -m venv venv"],
                "verifications": ["The virtual environment is created."],
            },
            {
                "phase": 3,
                "name": "Install dependencies",
                "interface": "CLI",
                "depends_on": [2],
                "operation_notes": ["venv/bin/pip install flask"],
                "verifications": ["Flask is installed in the virtual environment."],
            },
            {
                "phase": 4,
                "name": "Start API service",
                "interface": "CLI",
                "depends_on": [3],
                "operation_notes": ["venv/bin/python app.py &"],
                "verifications": ["A python process running app.py is present."],
            },
            {
                "phase": 5,
                "name": "Verify health check endpoint",
                "interface": "CLI",
                "depends_on": [4],
                "operation_notes": ["curl http://localhost:8080/health"],
                "verifications": ["The curl command returned a response."],
            },
        ],
    }

    class Reply:
        def invoke(self, messages: list) -> AIMessage:
            del messages
            return AIMessage(content=json.dumps(payload))

    current = [
        TestPhase(
            phase=1,
            name="Create API service code",
            interface="CODING",
            coding_operations=[
                CodingAction(action="create_file", file_path="app.py", content=flask, description="Flask application")
            ],
            verifications=["The file app.py contains a /health endpoint."],
        ),
        TestPhase(
            phase=2,
            name="Install dependencies",
            interface="CLI",
            depends_on=[1],
            operation_notes=["pip install flask"],
            verifications=["Flask is installed."],
        ),
    ]
    steps = [
        TestStep(
            step=1,
            interface="CODING",
            action="Create API service code",
            assertion="The file app.py contains a /health endpoint.",
            verifications=["The file app.py contains a /health endpoint."],
            coding_operations=current[0].coding_operations,
        )
    ]
    phases, _, reasoning = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").refine_plan(
        current,
        steps,
        "Update phase 2 to create and activate a virtual environment before install dependencies.",
        [],
    )
    assert reasoning == payload["reasoning"]
    assert [phase.operation_notes for phase in phases] == [
        [],
        ["python3 -m venv venv"],
        ["venv/bin/pip install flask"],
        ["venv/bin/python app.py &"],
        ["curl http://localhost:8080/health"],
    ]
    assert phases[0].coding_operations[0].content == flask
    assert [phase.depends_on for phase in phases] == [[], [1], [2], [3], [4]]


def test_prose_plan_reply_is_sent_back_for_valid_json() -> None:
    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner
    from aqe.state import TestPhase, TestStep

    valid = (
        '{"type": "plan_update", "reasoning": "Install inside a virtual environment.", "phases": ['
        '{"phase": 1, "name": "Install dependencies", "interface": "CLI", "depends_on": [], '
        '"operation_notes": ["python3 -m venv venv && venv/bin/pip install flask"], '
        '"verifications": ["Flask is installed in the virtual environment."]}]}'
    )

    class Reply:
        def __init__(self) -> None:
            self.calls = 0
            self.queries: list[str] = []

        def invoke(self, messages: list) -> AIMessage:
            self.calls += 1
            self.queries.append(messages[-1].content)
            return AIMessage(content="Phase 2 will be a command line phase running python3 -m venv venv.")

    current = [
        TestPhase(
            phase=1,
            name="Install dependencies",
            interface="CLI",
            operation_notes=["pip install flask"],
            verifications=["Flask is installed."],
        )
    ]
    steps = [
        TestStep(
            step=1,
            interface="CLI",
            action="pip install flask",
            assertion="Flask is installed.",
            verifications=["Flask is installed."],
            operation_notes=["pip install flask"],
        )
    ]
    model = Reply()
    phases, _, reasoning = ChatModelPlanner(model, "http://127.0.0.1:8765").refine_plan(
        current, steps, "Use a virtual environment.", []
    )
    assert model.calls == 1
    assert "not valid JSON" not in model.queries[0]
    assert phases == current
    assert reasoning.startswith("Failed to refine plan:")


def test_unparsed_json_is_requested_three_times() -> None:
    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner
    from aqe.state import TestPhase, TestStep

    class Reply:
        def __init__(self) -> None:
            self.calls = 0
            self.queries: list[str] = []

        def invoke(self, messages: list) -> AIMessage:
            self.calls += 1
            self.queries.append(messages[-1].content)
            return AIMessage(content="Create a virtual environment before installing.")

    current = [
        TestPhase(
            phase=1,
            name="Install dependencies",
            interface="CLI",
            operation_notes=["pip install flask"],
            verifications=["Flask is installed."],
        )
    ]
    steps = [
        TestStep(
            step=1,
            interface="CLI",
            action="pip install flask",
            assertion="Flask is installed.",
            verifications=["Flask is installed."],
            operation_notes=["pip install flask"],
        )
    ]
    model = Reply()
    phases, _, reasoning = ChatModelPlanner(model, "http://127.0.0.1:8765").refine_plan(
        current, steps, "Use a virtual environment.", []
    )
    assert model.calls == 1
    assert phases == current
    assert reasoning.startswith("Failed to refine plan:")


def test_decimal_phase_number_is_inserted_in_order() -> None:
    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner
    from aqe.state import TestPhase, TestStep

    reply = """
    {"type": "plan_update", "reasoning": "Inserted the install phase.", "phases": [
      {"phase": 1, "name": "Create API service code", "interface": "CODING",
       "coding_operations": [{"action": "create_file", "file_path": "app.py", "content": "from flask import Flask\\n", "description": "API"}],
       "verifications": ["The file app.py contains Flask."]},
      {"phase": 1.5, "name": "Install dependencies", "interface": "CLI", "depends_on": [1],
       "operation_notes": ["pip install flask"], "verifications": ["Flask is installed."]},
      {"phase": 2, "name": "Start API service", "interface": "CLI", "depends_on": [1.5],
       "operation_notes": ["python app.py &"], "verifications": ["The API process is present."]}
    ]}
    """

    class Reply:
        def invoke(self, messages: list) -> AIMessage:
            del messages
            return AIMessage(content=reply)

    current = [
        TestPhase(
            phase=1,
            name="Create API service code",
            interface="CODING",
            coding_operations=[],
            verifications=["The file app.py contains Flask."],
        ),
        TestPhase(
            phase=2,
            name="Start API service",
            interface="CLI",
            depends_on=[1],
            operation_notes=["python app.py &"],
            verifications=["The API process is present."],
        ),
    ]
    steps = [
        TestStep(
            step=1,
            interface="CODING",
            action="Create API service code",
            assertion="The file app.py contains Flask.",
            verifications=["The file app.py contains Flask."],
            coding_operations=[],
        )
    ]
    phases, _, reasoning = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").refine_plan(
        current, steps, "Before phase 2, you should install dependencies for the API app.", []
    )
    assert reasoning == "Inserted the install phase."
    assert [phase.name for phase in phases] == [
        "Create API service code",
        "Install dependencies",
        "Start API service",
    ]
    assert phases[1].operation_notes == ["pip install flask"]
    assert phases[2].depends_on == [2]


def test_cli_phase_keeps_the_bash_script() -> None:
    from aqe.llm import _coerce_phases, _phase_to_step

    script = "set -e\npython3 -m venv venv\nsource venv/bin/activate\npip install -r requirements.txt\n"
    phases, problem = _coerce_phases(
        [
            {
                "phase": 1,
                "name": "Install dependencies",
                "interface": "CLI",
                "script": script,
                "verifications": ["Dependencies are installed in the virtual environment."],
            }
        ]
    )
    assert problem is None
    assert phases[0].script == script
    step = _phase_to_step(phases[0])
    assert step.script == script
    assert step.interface == "CLI"


def test_script_with_raw_newlines_and_quotes_is_valid_json() -> None:
    from aqe.llm import _json_object

    broken = """
    {"type": "plan_update", "reasoning": "Create the virtual environment first.", "phases": [
      {"phase": 1, "name": "Create API", "interface": "CODING",
       "coding_operations": [{"action": "create_file", "file_path": "app.py",
         "content": "return {"status": "ok"}", "description": "API"}],
       "verifications": ["The file app.py exists."]},
      {"phase": 2, "name": "Create virtual environment", "interface": "CLI",
       "script": "python -m venv venv
source venv/bin/activate
",
       "verifications": ["The virtual environment is created."]}
    ]}
    """
    payload = _json_object(broken)
    script = payload["phases"][1]["script"]
    assert "python -m venv venv" in script
    assert "source venv/bin/activate" in script
    assert payload["phases"][0]["coding_operations"][0]["content"] == 'return {"status": "ok"}'


def test_a_plan_update_stores_the_model_file_content() -> None:
    from langchain_core.messages import AIMessage

    from aqe.llm import ChatModelPlanner
    from aqe.state import CodingAction, TestPhase, TestStep

    reply = (
        '{"type": "plan_update", "reasoning": "Create the virtual environment before install.", "phases": ['
        '{"phase": 1, "name": "Create API", "interface": "CODING", "depends_on": [], '
        '"coding_operations": [{"action": "create_file", "file_path": "app.py", '
        '"content": "from flask import Flask\\n", "description": "API"}], '
        '"verifications": ["The file app.py exists."]}, '
        '{"phase": 2, "name": "Create virtual environment", "interface": "CLI", "depends_on": [1], '
        '"script": "python3 -m venv venv\\n", '
        '"verifications": ["The virtual environment is created."]}, '
        '{"phase": 3, "name": "Install dependencies", "interface": "CLI", "depends_on": [2], '
        '"script": "source venv/bin/activate\\npip install -r requirements.txt\\n", '
        '"verifications": ["Dependencies are installed."]}]}'
    )

    class Reply:
        def invoke(self, messages: list) -> AIMessage:
            assert "Current phases JSON" in messages[-1].content
            return AIMessage(content=reply)

    flask = CodingAction(
        action="create_file",
        file_path="app.py",
        content="print('old')\n",
        description="API",
    )
    current = [
        TestPhase(
            phase=1,
            name="Create API",
            interface="CODING",
            coding_operations=[flask],
            verifications=["The file app.py exists."],
        ),
        TestPhase(
            phase=2,
            name="Install dependencies",
            interface="CLI",
            depends_on=[1],
            script="pip install -r requirements.txt\n",
            verifications=["Dependencies are installed."],
        ),
    ]
    steps = [
        TestStep(
            step=1,
            interface="CODING",
            action="Create API",
            assertion="The file app.py exists.",
            verifications=["The file app.py exists."],
            coding_operations=[flask],
        )
    ]
    phases, _, reasoning = ChatModelPlanner(Reply(), "http://127.0.0.1:8765").refine_plan(
        current,
        steps,
        "before install dependencies for the app, you should create and activate a virtual environment.",
        [],
    )
    assert reasoning == "Create the virtual environment before install."
    assert phases[0].coding_operations[0].content == "from flask import Flask\n"
    assert phases[1].name == "Create virtual environment"
    assert "python3 -m venv venv" in phases[1].script
    assert phases[2].name == "Install dependencies"

