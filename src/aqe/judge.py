"""Ask the chat model whether a page source satisfies a verification question."""

import json
import re
from typing import Protocol

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from aqe.llm import _json_object, _repair_json, _strip_fence, invoke_json
from aqe.state import Judgment

_SAMPLE_LIMIT = 4_000
_STDOUT_TERMS = re.compile(
    r"\b(stdout|outputs?|printed|prints)\b|standard output|"
    r"returns nothing|return nothing|empty content|empty output|prints nothing|no output|nothing is returned",
    re.IGNORECASE,
)


def mentions_stdout(question: str) -> bool:
    """Stdout content belongs in the check only when the assertion talks about output."""
    return _STDOUT_TERMS.search(question) is not None


_JUDGE_SYSTEM = (
    "You judge whether the supplied text satisfies one verification question. "
    "The text may be an HTML page, the command streams the verification asks about, "
    "or the files a coding phase wrote. "
    "Use only that text. Do not assume content that is not in it. "
    "A page check uses the page text. "
    "A coding check uses the file path and the file content. "
    "A message that a coding operation succeeded does not satisfy a check about what a file contains. "
    "If the file is missing or the operation reports an error, a check about that file's content fails. "
    "A CLI check reaches you only after $? is 0. A non-zero $? already failed the command. "
    "Judge the verification from stdout and stderr. Both streams are in the text. "
    "A script result, when present, records VIRTUAL_ENV, venv_dir, and venv_python from that same shell. "
    "VIRTUAL_ENV set to a path means that shell activated a virtual environment. "
    "venv_dir=present means a directory named venv exists. "
    "Use that script result for a check about creating or activating a virtual environment. "
    "Do not require stdout to be non-empty unless the verification says so. "
    "When a stream is absent, ignore that stream. "
    "Read the assertion as a full statement of the original check. "
    "If the assertion says the command returns nothing or empty content, pass when stdout is empty. "
    "If the assertion says standard output or stdout is empty, pass only when stdout is empty. "
    "If the assertion says standard error or stderr is empty, pass only when stderr is empty. "
    "stdout length and stderr length are exact. Length 0 means that stream is empty. "
    "If the assertion says stdout is not empty, pass that part only when stdout length is greater than 0. "
    "The question may be a sentence or a check such as contains:TEXT. "
    "Reply with one JSON object and no other text. "
    'Keys: "passed" (true or false) and "judgment" '
    "(one or two sentences that quote the relevant text and say why the question passes or fails)."
)


class PageJudge(Protocol):
    def judge(self, question: str, page_source: str) -> Judgment:
        """Return the model's pass or fail decision and its explanation."""


_PASSED_FIELD = re.compile(r'"passed"\s*:\s*(true|false|"true"|"false")', re.IGNORECASE)
_JUDGMENT_FIELD = re.compile(r'"judgment"\s*:\s*"(.*?)"\s*(?:,|\})', re.IGNORECASE | re.DOTALL)


def _loose_judgment(content: str) -> dict | None:
    """Read a decision when the model almost returned the JSON object."""
    passed = _PASSED_FIELD.search(content)
    if not passed:
        return None
    judgment = _JUDGMENT_FIELD.search(content)
    text = judgment.group(1).strip() if judgment else " ".join(content.split())
    text = text.replace('\\"', '"').replace("\\n", " ").strip()
    if not text:
        return None
    return {"passed": passed.group(1).strip('"').lower() == "true", "judgment": text[:1000]}


_DECISION_KEYS = ("passed", "pass", "result", "success", "verdict")
_EXPLANATION_KEYS = ("judgment", "explanation", "reason", "message")


def _sample(text: str, limit: int = _SAMPLE_LIMIT) -> str:
    if len(text) <= limit:
        return text
    head = limit * 3 // 4
    tail = limit - head
    omitted = len(text) - limit
    return f"{text[:head]}\n...[truncated {omitted} characters]...\n{text[-tail:]}"


def _split_exit_code(page_source: str) -> tuple[str, str | None]:
    """Separate a trailing exit code so it is not counted as stderr."""
    match = re.search(r"\n\nexit_code:\s*(\S+)\s*$", page_source)
    if match is None:
        return page_source, None
    return page_source[: match.start()], match.group(1)


def _split_script_result(page_source: str) -> tuple[str, str | None]:
    marker = "\n\nscript_result:\n"
    if marker not in page_source:
        return page_source, None
    head, result = page_source.rsplit(marker, 1)
    return head, result.strip() or None


def present_for_judge(page_source: str, question: str = "") -> str:
    """Show stdout and stderr. $? is decided before this text is judged."""
    del question
    marker = "\n\nstderr:\n"
    page_source, script_result = _split_script_result(page_source)
    page_source, _exit_code = _split_exit_code(page_source)

    if page_source.startswith("stdout:") and marker in page_source:
        stdout, stderr = page_source.rsplit(marker, 1)
        stdout = stdout.removeprefix("stdout:\n").removeprefix("stdout:")
        parts = [
            f"stdout length: {len(stdout)} characters\nstdout:\n{_sample(stdout)}",
            f"stderr length: {len(stderr)} characters\nstderr:\n{_sample(stderr, 2_000)}",
        ]
        if script_result:
            parts.append(f"script result:\n{script_result}")
        return "\n\n".join(parts)

    return _sample(page_source.strip(), 8_000)


def _objects(content: str) -> list[dict]:
    text = _strip_fence(content)
    found: list[dict] = []
    start = 0
    while True:
        index = text.find("{", start)
        if index < 0:
            return found
        blob = text[index:]
        try:
            value, _ = json.JSONDecoder().raw_decode(blob)
        except json.JSONDecodeError:
            try:
                value, _ = json.JSONDecoder().raw_decode(_repair_json(blob))
            except json.JSONDecodeError:
                start = index + 1
                continue
        if isinstance(value, dict):
            found.append(value)
        start = index + 1


def _as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        folded = value.strip().lower()
        if folded in {"true", "yes", "pass", "passed"}:
            return True
        if folded in {"false", "no", "fail", "failed"}:
            return False
    return None


def _decision_payload(content: str) -> dict | None:
    chosen = None
    for payload in _objects(content):
        if any(key in payload for key in _DECISION_KEYS):
            chosen = payload
            break
        chosen = chosen or payload
    if chosen and any(key in chosen for key in _DECISION_KEYS):
        return chosen
    return _loose_judgment(content)


def parse_judgment(content: str) -> Judgment:
    """Read the model's JSON decision."""
    payload = _decision_payload(content)
    if payload is None:
        snippet = " ".join(content.split())[:300] or "the model returned an empty reply"
        raise ValueError(f"verification judgment was not valid JSON: {snippet}")
    passed = None
    for key in _DECISION_KEYS:
        if key in payload:
            passed = _as_bool(payload[key])
            if passed is not None:
                break
    if passed is None:
        raise ValueError("verification judgment did not include passed")
    text = ""
    for key in _EXPLANATION_KEYS:
        text = str(payload.get(key) or "").strip()
        if text:
            break
    if not text:
        raise ValueError("verification judgment did not include an explanation")
    return Judgment(passed=passed, judgment=text)


@dataclass
class CliCheck:
    """A CLI verification decision, or a command the agent should run before deciding."""

    passed: bool | None
    judgment: str
    script: str = ""


_CLI_VERIFY_SYSTEM = (
    "You verify one CLI check. "
    "You receive the verification statement, the value of $?, stdout, and stderr from the phase script. "
    "Reply with one JSON object and nothing else. "
    "If those four items are enough to decide, return "
    '{"type": "decision", "passed": true or false, "judgment": "one or two sentences"}. '
    "Use $? , stdout, and stderr only. Do not assume a file, process, or directory that is not shown there. "
    "Pass only when the statement's claim about $? , stdout, or stderr is visible in those values. "
    "A vague statement such as 'the background service is running' fails when those values do not show it. "
    "If the statement is about something those streams do not show, such as the content of a log file, return "
    '{"type": "verify", "script": "a bash script that prints the evidence", "judgment": "what this command checks"}. '
    "The script runs in the same directory as the phase script. "
    "It must print the evidence on stdout. A failing command may exit non-zero."
)

_CLI_VERIFY_FOLLOWUP = (
    "You verify one CLI check from the command you requested. "
    "You receive the verification statement, the value of $?, stdout, and stderr from that command. "
    "Reply with one JSON object and nothing else: "
    '{"type": "decision", "passed": true or false, "judgment": "one or two sentences"}. '
    "Decide from $? , stdout, and stderr. Do not request another command."
)


def parse_cli_check(content: str) -> CliCheck:
    """Read a CLI decision or a verification command."""
    payload = _json_object(content)
    kind = str(payload.get("type") or "").strip().lower()
    script = str(payload.get("script") or payload.get("instruction") or "").strip()
    judgment = str(payload.get("judgment") or payload.get("reason") or "").strip()
    if kind == "verify" or (script and kind != "decision"):
        if not script:
            raise ValueError("verification instruction did not include a script")
        return CliCheck(passed=None, judgment=judgment or "Run the verification command.", script=script)
    passed = _as_bool(payload.get("passed"))
    if passed is None:
        raise ValueError("verification decision did not include passed")
    if not judgment:
        raise ValueError("verification decision did not include a judgment")
    return CliCheck(passed=passed, judgment=judgment)


class ChatPageJudge:
    """Send the page source and the verification question to the configured model."""

    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def judge(self, question: str, page_source: str) -> Judgment:
        source = present_for_judge(page_source, question)
        messages = [
            SystemMessage(content=_JUDGE_SYSTEM),
            HumanMessage(
                content=f"Verification question:\n{question.strip()}\n\nText:\n{source}"
            ),
        ]
        judgment, _content = invoke_json(self.model, messages, parse_judgment)
        return judgment

    def judge_cli(
        self,
        question: str,
        exit_code: int,
        stdout: str,
        stderr: str,
        *,
        follow_up: bool = False,
    ) -> CliCheck:
        """Decide a CLI check, or request one command that gathers the missing evidence."""
        system = _CLI_VERIFY_FOLLOWUP if follow_up else _CLI_VERIFY_SYSTEM
        messages = [
            SystemMessage(content=system),
            HumanMessage(
                content=(
                    f"Verification statement:\n{question.strip()}\n\n"
                    f"$?: {exit_code}\n\n"
                    f"stdout:\n{stdout}\n\n"
                    f"stderr:\n{stderr}\n"
                )
            ),
        ]
        check, _content = invoke_json(self.model, messages, parse_cli_check)
        if follow_up and check.script:
            return CliCheck(
                passed=False,
                judgment=check.judgment or "The model did not decide the verification.",
            )
        return check


def build_judge() -> ChatPageJudge:
    from aqe.chat import get_chat_model

    return ChatPageJudge(get_chat_model())


_STDERR_JUDGE_SYSTEM = (
    "You judge whether stderr output from a command execution represents an actual error or just a warning/notification. "
    "An actual error means the command failed to accomplish its intended purpose. "
    "Warnings, deprecation notices, informational messages, and suggestions are not errors. "
    "Consider the context: exit code 0 with warnings is usually not an error. "
    "Reply with one JSON object and no other text. "
    'Keys: "is_error" (true or false) and "explanation" '
    "(one sentence explaining why this is or is not an error)."
)


def _stderr_decision(content: str) -> dict:
    """Read a stderr judgment, including a reply that is almost JSON."""
    text = _strip_fence(content)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = _decision_payload(content)
    if not isinstance(payload, dict):
        raise ValueError("stderr judgment was not valid JSON")
    return payload


def judge_stderr(model: BaseChatModel, stderr: str, exit_code: int) -> tuple[bool, str]:
    """Judge whether stderr content represents an actual error or just a warning/notification."""
    if not stderr.strip():
        return False, "No stderr output"

    messages = [
        SystemMessage(content=_STDERR_JUDGE_SYSTEM),
        HumanMessage(content=f"Exit code: {exit_code}\n\nstderr output:\n{stderr}"),
    ]
    try:
        payload, _content = invoke_json(model, messages, _stderr_decision)
    except ValueError as exc:
        return True, f"Unable to parse stderr judgment: {exc}"

    is_error = None
    for key in ("is_error", "error", "failed"):
        if key in payload:
            is_error = _as_bool(payload[key])
            if is_error is not None:
                break
    if is_error is None:
        return True, "Stderr judgment did not include error decision"
    explanation = ""
    for key in ("explanation", "judgment", "reason"):
        explanation = str(payload.get(key) or "").strip()
        if explanation:
            break
    return is_error, explanation or "No explanation provided"
