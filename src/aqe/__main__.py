"""aqe run and aqe serve."""

from __future__ import annotations

import argparse
import sys
import threading
import uuid
from pathlib import Path

import uvicorn

from aqe.api import create_app
from aqe.config import EngineConfig
from aqe.errors import SpecValidationError
from aqe.service import RunService
from aqe.settings import get_settings
from aqe.specs import load_spec


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _start_sut(webhook_path: Path, port: int) -> None:
    root = _repo_root()
    sut_path = root / "examples" / "sut" / "server.py"
    if not sut_path.is_file():
        sut_path = Path.cwd() / "examples" / "sut" / "server.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("aqe_sut", sut_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load the sample system under test at {sut_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    server = module.serve(webhook_path, port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()


def _run() -> int:
    settings = get_settings()
    if not settings.spec_path:
        print("SPEC_PATH is required in .env", file=sys.stderr)
        return 2
    config = EngineConfig.from_env()
    specification = load_spec(Path(settings.spec_path))
    service = RunService(config)
    run_id = uuid.uuid4().hex
    evidence = service.run_dir(run_id) / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    config = config.model_copy(update={"target_url": f"http://127.0.0.1:{settings.sut_port}"})
    service.config = config
    _start_sut(evidence / "webhook.json", settings.sut_port)
    try:
        service.submit(specification, run_id=run_id)
    except SpecValidationError as exc:
        print(exc.message, file=sys.stderr)
        return 2
    snapshot = service.wait(run_id)
    report_path = service.run_dir(run_id) / "report.json"
    print(report_path)
    verdict = (snapshot.get("report") or {}).get("verdict")
    if verdict != "pass":
        reason = (snapshot.get("report") or {}).get("reason")
        if reason:
            print(reason, file=sys.stderr)
        return 1
    return 0


def _serve() -> int:
    settings = get_settings()
    config = EngineConfig.from_env()
    service = RunService(config)
    public_url = f"http://{settings.host}:{settings.port}"
    app = create_app(service, public_url=public_url)
    uvicorn.run(app, host=settings.host, port=settings.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aqe")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="Execute the specification in SPEC_PATH")
    sub.add_parser("serve", help="Serve the API, UI, and A2A agent")
    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve()
    return _run()


if __name__ == "__main__":
    raise SystemExit(main())
