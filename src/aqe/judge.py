"""Ask the chat model whether a page source satisfies a verification question."""

import json
import re
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from aqe.llm import _repair_json, _strip_fence
from aqe.state import Judgment

_SAMPLE_LIMIT = 4_000
_STDERR_TERMS = re.compile(
    r"\b(errors?|issues?|exceptions?|stderr|traceback|warnings?|failures?|failed|stack traces?)\b|standard error",
    re.IGNORECASE,
)


def mentions_stderr(question: str) -> bool:
    """Stderr belongs in the check only when the assertion talks about a problem."""
    return _STDERR_TERMS.search(question) is not None


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
    "The text may be an HTML page, or the command streams the verification asks about. "
    "Use only that text. Do not assume content that is not in it. "
    "A page check uses the page text. "
    "stdout is included only when the verification mentions stdout, standard output, or the command output. "
    "stderr is included only when the verification mentions an error, issue, exception, warning, failure, traceback, or stderr. "
    "Do not require stdout to be non-empty unless the verification says so. "
    "When a stream is absent, ignore that stream. "
    "Read the assertion as a full statement of the original check. "
    "Do not narrow it to an exit code unless the statement is about the exit code. "
    "If the assertion says the command returns nothing or empty content, pass when stdout is empty. "
    "If the assertion says standard output or stdout is empty, pass only when stdout is empty. "
    "If the assertion says standard error or stderr is empty, pass only when stderr is empty. "
    "stdout length and stderr length are exact. Length 0 means that stream is empty. "
    "If the assertion says stdout is not empty, pass that part only when stdout length is greater than 0. "
    "IMPORTANT: For CLI commands, the exit_code is the definitive indicator of success or failure. "
    "If exit_code is non-zero, the command FAILED regardless of stderr content. "
    "Only analyze stderr/stdout for content-based checks when exit_code is 0. "
    "IMPORTANT: Python errors are ALWAYS errors, not warnings. "
    "If the text contains Python errors like ModuleNotFoundError, ImportError, NameError, SyntaxError, TypeError, ValueError, KeyError, AttributeError, or any other Python exception, the verification FAILED. "
    "Python tracebacks starting with 'Traceback (most recent call last):' are always errors. "
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


def present_for_judge(page_source: str, question: str = "") -> str:
    """Show a command stream only when the assertion mentions that stream."""
    marker = "\n\nstderr:\n"
    
    # Check if exit_code is present in the evidence
    exit_code_marker = "\nexit_code:"
    exit_code = None
    if exit_code_marker in page_source:
        parts = page_source.split(exit_code_marker)
        if len(parts) > 1:
            exit_code = parts[1].strip().split()[0] if parts[1].strip() else None
            # Remove exit_code from page_source for further processing
            page_source = parts[0]
    
    if page_source.startswith("stdout:") and marker in page_source:
        stdout, stderr = page_source.rsplit(marker, 1)
        stdout = stdout.removeprefix("stdout:\n").removeprefix("stdout:")
        parts: list[str] = []
        
        # Always include exit code if available
        if exit_code is not None:
            parts.append(f"exit_code: {exit_code}")
        
        if mentions_stdout(question):
            parts.append(f"stdout length: {len(stdout)} characters\nstdout:\n{_sample(stdout)}")
        if mentions_stderr(question):
            parts.append(f"stderr length: {len(stderr)} characters\nstderr:\n{_sample(stderr, 2_000)}")
        if not parts:
            return "The command finished. stdout and stderr are not part of this check."
        return "\n\n".join(parts)
    
    # For non-stdout/stderr format, include exit code if available
    if exit_code is not None:
        return f"exit_code: {exit_code}\n\n{_sample(page_source.strip(), 8_000)}"
    
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


class ChatPageJudge:
    """Send the page source and the verification question to the configured model."""

    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def judge(self, question: str, page_source: str) -> Judgment:
        source = present_for_judge(page_source, question)
        message = self.model.invoke(
            [
                SystemMessage(content=_JUDGE_SYSTEM),
                HumanMessage(
                    content=f"Verification question:\n{question.strip()}\n\nText:\n{source}"
                ),
            ]
        )
        content = message.content if isinstance(message.content, str) else str(message.content)
        try:
            return parse_judgment(content)
        except ValueError:
            retry = self.model.invoke(
                [
                    SystemMessage(content=_JUDGE_SYSTEM),
                    HumanMessage(
                        content=(
                            f"Verification question:\n{question.strip()}\n\nText:\n{source}\n\n"
                            "Reply with only one JSON object. "
                            'Keys are "passed" (true or false) and "judgment" (one or two sentences).'
                        )
                    ),
                ]
            )
            retried = retry.content if isinstance(retry.content, str) else str(retry.content)
            return parse_judgment(retried)


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


def judge_stderr(model: BaseChatModel, stderr: str, exit_code: int) -> tuple[bool, str]:
    """Judge whether stderr content represents an actual error or just a warning/notification."""
    if not stderr.strip():
        return False, "No stderr output"
    
    message = model.invoke(
        [
            SystemMessage(content=_STDERR_JUDGE_SYSTEM),
            HumanMessage(
                content=f"Exit code: {exit_code}\n\nstderr output:\n{stderr}"
            ),
        ]
    )
    content = message.content if isinstance(message.content, str) else str(message.content)
    
    try:
        # Try to parse JSON directly for stderr judgment
        text = _strip_fence(content)
        payload = json.loads(text)
        
        if not isinstance(payload, dict):
            return True, f"Stderr judgment was not a JSON object: {content[:200]}"
        
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
        
        if not explanation:
            explanation = "No explanation provided"
        
        return is_error, explanation
    except json.JSONDecodeError:
        # Fallback to the general payload parser if direct JSON parsing fails
        try:
            payload = _decision_payload(content)
            if payload is None:
                return True, f"Unable to parse stderr judgment: {content[:200]}"
            
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
            
            if not explanation:
                explanation = "No explanation provided"
            
            return is_error, explanation
        except (ValueError, KeyError) as e:
            return True, f"Error judging stderr: {e!s}"
    except (ValueError, KeyError) as e:
        return True, f"Error judging stderr: {e!s}"
