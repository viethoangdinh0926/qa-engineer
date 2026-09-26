"""Live browser checks. Skipped when the host cannot run them."""

import socket
import threading
from pathlib import Path

import pytest

from aqe.capabilities import probe_host
from aqe.config import EngineConfig
from aqe.service import RunService
from aqe.specs import load_spec

SAMPLE = Path("examples/specs/registration.md")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.integration
def test_live_registration(tmp_path: Path) -> None:
    config = EngineConfig(runs_dir=tmp_path / "runs")
    capabilities = probe_host(config)
    if not capabilities.browser.available:
        pytest.skip(
            capabilities.browser.detail or "browser driver is unavailable"
        )
    import importlib.util

    sut_path = Path("examples/sut/server.py").resolve()
    spec = importlib.util.spec_from_file_location("aqe_sut", sut_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    run_id = "live-registration"
    evidence = config.runs_dir / run_id / "evidence"
    evidence.mkdir(parents=True)
    port = _free_port()
    server = module.serve(evidence / "webhook.json", port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = config.model_copy(update={"target_url": f"http://127.0.0.1:{port}"})
    service = RunService(config, probe=lambda: probe_host(config))
    try:
        service.submit(load_spec(SAMPLE), run_id=run_id)
        finished = service.wait(run_id, timeout=60)
    finally:
        server.shutdown()
    assert finished["report"]["verdict"] == "pass"
    assert finished["report"]["specification"] == load_spec(SAMPLE)
