"""Product HTTP API, server-sent events, the UI, and the A2A routes."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.fastapi_routes import add_a2a_routes_to_fastapi
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.routes.rest_routes import create_rest_routes
from a2a.server.tasks import InMemoryTaskStore

from aqe.a2a_agent import ImmediateRequestHandler, QualityAgentExecutor, build_agent_card
from aqe.config import EngineConfig
from aqe.errors import SpecValidationError
from aqe.service import RunService

UI_DIR = Path(__file__).resolve().parent / "ui"


class RunRequest(BaseModel):
    specification: str = ""


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
            snapshot = engine.submit(body.specification)
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
