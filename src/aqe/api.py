"""Product HTTP API, server-sent events, the UI, and the A2A routes."""

from __future__ import annotations

import json
from pathlib import Path

from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.fastapi_routes import add_a2a_routes_to_fastapi
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.routes.rest_routes import create_rest_routes
from a2a.server.tasks import InMemoryTaskStore
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from aqe.a2a_agent import (
    ImmediateRequestHandler,
    QualityAgentExecutor,
    build_agent_card,
)
from aqe.config import EngineConfig
from aqe.errors import SpecValidationError
from aqe.plan_storage import PlanStorageManager
from aqe.service import RunService
from aqe.state import PlannerChatHistory

UI_DIR = Path(__file__).resolve().parent / "ui"


class RunRequest(BaseModel):
    specification: str = ""
    wait_for_approval: bool = True


def create_app(service: RunService | None = None, *, public_url: str = "http://127.0.0.1:8000") -> FastAPI:
    engine = service or RunService(EngineConfig())
    app = FastAPI(title="aqe")
    app.state.service = engine

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/capabilities")
    def capabilities() -> dict[str, object]:
        return engine.probe().as_public()

    @app.post("/v1/runs", status_code=202)
    def create_run(body: RunRequest):
        try:
            wait_for_approval = body.model_dump().get("wait_for_approval", True)
            snapshot = engine.submit(body.specification, wait_for_approval=wait_for_approval)
        except SpecValidationError as exc:
            return JSONResponse({"error": exc.message}, status_code=400)
        return {"id": snapshot["id"], "status": snapshot["status"]}

    @app.get("/v1/runs")
    def list_runs() -> dict[str, list]:
        return {"runs": engine.list_runs()}

    @app.get("/v1/runs/{run_id}")
    def get_run(run_id: str):
        snapshot = engine.get(run_id)
        if snapshot is None:
            return JSONResponse({"error": "Run not found."}, status_code=404)
        return snapshot

    @app.post("/v1/runs/{run_id}:cancel")
    def cancel_run(run_id: str):
        snapshot = engine.cancel(run_id)
        if snapshot is None:
            return JSONResponse({"error": "Run not found."}, status_code=404)
        return snapshot

    @app.post("/v1/runs/{run_id}:start")
    def start_run(run_id: str):
        """Start execution for a run that was waiting for plan approval."""
        snapshot = engine.start_execution(run_id)
        if snapshot is None:
            return JSONResponse({"error": "Run not found."}, status_code=404)
        if "error" in snapshot:
            return JSONResponse(snapshot, status_code=400)
        return snapshot

    @app.get("/v1/runs/{run_id}/events")
    def run_events(run_id: str):
        if engine.get(run_id) is None:
            return JSONResponse({"error": "Run not found."}, status_code=404)

        def generate():
            stream = engine.iter_snapshots(run_id)
            assert stream is not None
            for snapshot in stream:
                yield f"data: {json.dumps(snapshot)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # Planner chat endpoints
    @app.post("/v1/runs/{run_id}/planner/chat")
    def planner_chat(run_id: str, body: dict):
        """Send a message to the planner and get a response."""
        if engine.get(run_id) is None:
            return JSONResponse({"error": "Run not found."}, status_code=404)
        
        message = body.get("message", "")
        if not message:
            return JSONResponse({"error": "Message is required."}, status_code=400)
        
        # Get or create chat history
        plan_storage = PlanStorageManager(engine.config.runs_dir)
        chat = plan_storage.load_chat_history(run_id) or PlannerChatHistory(run_id=run_id)
        
        # Add user message
        chat.add_message("user", message)
        
        # Get current plan
        plan = plan_storage.load_plan(run_id)
        if not plan:
            chat.add_message("assistant", "No plan exists yet. Please create a run first.")
            plan_storage.save_chat_history(chat)
            return {"messages": [msg.model_dump(mode="json") for msg in chat.messages]}
        
        # Get current plan phases and steps
        phases, steps = plan.phases, plan.steps
        
        # Get planner and refine plan
        planner = engine.planner
        if hasattr(planner, 'refine_plan'):
            chat_history = [msg.model_dump(mode="json") for msg in chat.messages]
            refined_phases, refined_steps, reasoning = planner.refine_plan(
                phases, steps, message, chat_history
            )
            
            # Check if the plan was actually updated
            if refined_phases == phases and refined_steps == steps:
                # No changes made - likely just an answer to a question
                chat.add_message("assistant", reasoning)
            else:
                # Plan was updated - directly set the new phases and steps
                plan.phases = refined_phases
                plan.steps = refined_steps
                plan_storage.save_plan(plan)
                chat.add_message("assistant", f"Plan updated. {reasoning}")
        else:
            chat.add_message("assistant", "Planner does not support plan refinement.")
        
        # Save chat history
        plan_storage.save_chat_history(chat)
        
        return {"messages": [msg.model_dump(mode="json") for msg in chat.messages]}

    @app.get("/v1/runs/{run_id}/planner/chat/history")
    def get_planner_chat_history(run_id: str):
        """Get the planner chat history."""
        if engine.get(run_id) is None:
            return JSONResponse({"error": "Run not found."}, status_code=404)
        
        plan_storage = PlanStorageManager(engine.config.runs_dir)
        chat = plan_storage.load_chat_history(run_id)
        if not chat:
            return {"messages": []}
        
        return {"messages": [msg.model_dump(mode="json") for msg in chat.messages]}

    # Plan management endpoints
    @app.get("/v1/runs/{run_id}/plan")
    def get_plan(run_id: str):
        """Get the current plan for a run."""
        if engine.get(run_id) is None:
            return JSONResponse({"error": "Run not found."}, status_code=404)
        
        plan_storage = PlanStorageManager(engine.config.runs_dir)
        plan = plan_storage.load_plan(run_id)
        if not plan:
            return JSONResponse({"error": "Plan not found."}, status_code=404)
        
        return plan.model_dump(mode="json")

    @app.put("/v1/runs/{run_id}/plan")
    def update_plan(run_id: str, body: dict):
        """Update the plan (if editing is allowed)."""
        if engine.get(run_id) is None:
            return JSONResponse({"error": "Run not found."}, status_code=404)
        
        plan_storage = PlanStorageManager(engine.config.runs_dir)
        plan = plan_storage.load_plan(run_id)
        if not plan:
            return JSONResponse({"error": "Plan not found."}, status_code=404)
        
        # Only allow updates if plan is in draft status
        if plan.status != "draft":
            return JSONResponse({"error": "Cannot update plan that is not in draft status."}, status_code=400)
        
        # TODO: Implement plan update logic
        # For now, return the current plan
        return plan.model_dump(mode="json")

    @app.post("/v1/runs/{run_id}/plan/approve")
    def approve_plan(run_id: str, body: dict):
        """Approve the plan for execution."""
        if engine.get(run_id) is None:
            return JSONResponse({"error": "Run not found."}, status_code=404)
        
        plan_storage = PlanStorageManager(engine.config.runs_dir)
        plan = plan_storage.load_plan(run_id)
        if not plan:
            return JSONResponse({"error": "Plan not found."}, status_code=404)
        
        # If plan is already approved, cancel current execution and restart
        if plan.status == "approved":
            # Cancel current execution if running
            engine.cancel(run_id)
        
        # Update plan status
        plan.status = "approved"
        plan.approved_by = body.get("user", "unknown")
        plan_storage.save_plan(plan)
        
        # Start execution
        engine.start_execution(run_id)
        
        return plan.model_dump(mode="json")

    @app.post("/v1/runs/{run_id}/plan/reject")
    def reject_plan(run_id: str, body: dict):
        """Reject the plan."""
        if engine.get(run_id) is None:
            return JSONResponse({"error": "Run not found."}, status_code=404)
        
        plan_storage = PlanStorageManager(engine.config.runs_dir)
        plan = plan_storage.load_plan(run_id)
        if not plan:
            return JSONResponse({"error": "Plan not found."}, status_code=404)
        
        if plan.status != "draft":
            return JSONResponse({"error": "Plan is not in draft status."}, status_code=400)
        
        # Update plan status
        plan.status = "rejected"
        plan.rejection_reason = body.get("reason", "No reason provided")
        plan_storage.save_plan(plan)
        
        return plan.model_dump(mode="json")

    @app.get("/v1/runs/{run_id}/plan/download")
    def download_plan(run_id: str, format: str = "json"):
        """Download the plan in the specified format."""
        if engine.get(run_id) is None:
            return JSONResponse({"error": "Run not found."}, status_code=404)
        
        plan_storage = PlanStorageManager(engine.config.runs_dir)
        plan = plan_storage.load_plan(run_id)
        if not plan:
            return JSONResponse({"error": "Plan not found."}, status_code=404)
        
        if format == "json":
            return JSONResponse(plan.model_dump(mode="json"))
        elif format == "markdown":
            # TODO: Convert plan to markdown format
            return JSONResponse({"error": "Markdown format not yet implemented."}, status_code=501)
        else:
            return JSONResponse({"error": f"Unsupported format: {format}"}, status_code=400)

    @app.get("/")
    def home() -> FileResponse:
        return FileResponse(UI_DIR / "index.html")

    @app.get("/runs/{run_id}")
    def run_page(run_id: str) -> FileResponse:
        del run_id
        return FileResponse(UI_DIR / "index.html")

    @app.get("/static/{asset}")
    def static_asset(asset: str):
        path = (UI_DIR / asset).resolve()
        if not path.is_relative_to(UI_DIR.resolve()) or not path.is_file():
            return JSONResponse({"error": "Not found."}, status_code=404)
        return FileResponse(path)

    card = build_agent_card(public_url.rstrip("/") + "/")
    handler = ImmediateRequestHandler(
        agent_executor=QualityAgentExecutor(engine),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url="/"),
        rest_routes=create_rest_routes(handler),
    )
    return app
