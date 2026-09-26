"""A2A agent surface. The task id is the run id."""

from __future__ import annotations

import asyncio

from a2a.helpers.proto_helpers import new_data_part, new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import TaskUpdater
from a2a.types.a2a_pb2 import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    SendMessageRequest,
)
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT
from a2a.utils.errors import InvalidParamsError
from google.protobuf.json_format import MessageToDict

from aqe.errors import SpecValidationError
from aqe.service import RunService

SKILL_DESCRIPTION = (
    "Execute a quality-engineering specification. "
    "Read GET /v1/capabilities before sending a specification. "
    "Poll GetTask until status.state is TASK_STATE_COMPLETED, TASK_STATE_FAILED, "
    "TASK_STATE_REJECTED, or TASK_STATE_CANCELED, then read the artifact."
)


def build_agent_card(url: str) -> AgentCard:
    return AgentCard(
        name="aqe",
        description="Agentic quality engineering. Submit a test specification and poll for the JSON report.",
        version="0.1.0",
        supported_interfaces=[
            AgentInterface(
                url=url,
                protocol_binding="JSONRPC",
                protocol_version=PROTOCOL_VERSION_CURRENT,
            ),
            AgentInterface(
                url=url,
                protocol_binding="HTTP+JSON",
                protocol_version=PROTOCOL_VERSION_CURRENT,
            ),
        ],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain", "text/markdown", "application/octet-stream"],
        default_output_modes=["application/json"],
        skills=[
            AgentSkill(
                id="execute-quality-spec",
                name="Execute quality spec",
                description=SKILL_DESCRIPTION,
                tags=["quality", "testing"],
                input_modes=["text/plain", "text/markdown"],
                output_modes=["application/json"],
                examples=["Verify that user registration triggers a webhook."],
            )
        ],
    )


def specification_from_message(message) -> str:
    chunks: list[str] = []
    for part in message.parts:
        kind = part.WhichOneof("content")
        if kind == "text" and part.text.strip():
            chunks.append(part.text)
        elif kind == "raw" and part.raw:
            chunks.append(bytes(part.raw).decode("utf-8"))
        elif kind == "data":
            data = MessageToDict(part.data)
            if isinstance(data, str) and data.strip():
                chunks.append(data)
    return "\n".join(chunks).strip()


class ImmediateRequestHandler(DefaultRequestHandler):
    """Return SendMessage as soon as the task is working, and reject empty specs."""

    async def on_message_send(self, params: SendMessageRequest, context):
        if not specification_from_message(params.message):
            raise InvalidParamsError(
                message="A specification text or file part is required."
            )
        if params.message.task_id:
            existing = await self.task_store.get(params.message.task_id, context)
            if existing is not None:
                raise InvalidParamsError(
                    message="A second message on the same task is not accepted."
                )
        params.configuration.return_immediately = True
        return await super().on_message_send(params, context)


class QualityAgentExecutor(AgentExecutor):
    def __init__(self, service: RunService) -> None:
        self.service = service

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.current_task is not None:
            raise InvalidParamsError(
                message="A second message on the same task is not accepted."
            )
        message = context.message
        specification = specification_from_message(message) if message else ""
        if not specification:
            raise InvalidParamsError(
                message="A specification text or file part is required."
            )
        task_id = context.task_id or ""
        context_id = context.context_id or ""
        if message is not None:
            if task_id:
                message.task_id = task_id
            if context_id:
                message.context_id = context_id
            await event_queue.enqueue_event(new_task_from_user_message(message))
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.start_work()
        try:
            self.service.submit(specification, run_id=task_id)
        except SpecValidationError as exc:
            raise InvalidParamsError(message=exc.message) from exc
        snapshot = await asyncio.to_thread(self.service.wait, task_id)
        report = snapshot.get("report") or {}
        await updater.add_artifact(
            [new_data_part(report, media_type="application/json")],
            name="report.json",
        )
        verdict = report.get("verdict")
        if verdict in {"pass", "fail"}:
            await updater.complete()
        elif verdict == "error":
            await updater.failed()
        elif verdict == "canceled":
            await updater.cancel()
        else:
            await updater.reject()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.task_id:
            self.service.cancel(context.task_id)
        updater = TaskUpdater(event_queue, context.task_id or "", context.context_id or "")
        try:
            await updater.cancel()
        except RuntimeError:
            return
