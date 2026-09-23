# Agentic Quality Engineering
`aqe` is a quality-engineering service that turns a specification into a test plan, runs each step through a browser, desktop, or sandboxed CLI adapter, and returns one JSON report.
A run moves through plan, capability check, execute, validate, and reflect. The same run is available to a person in the browser, to a script over HTTP, and to another agent over [A2A](https://github.com/a2aproject/A2A).
## Install
Python 3.11 or newer.
```bash
uv venv
uv pip install -e .
uv run playwright install chromium
uv run playwright install-deps
docker build -t aqe-sandbox:local docker/sandbox
```
That install includes Playwright, the desktop driver packages, the OpenAI, Anthropic, and Ollama clients, and pytest. Chromium and the sandbox image are downloaded by the commands above. Copy `.env_template` to `.env` and set `LLM_PROVIDER` to `openai`, `anthropic`, or `ollama`, plus `LLM_MODEL` and the matching credentials. OpenAI can use `OPENAI_API_KEY`, an AIA gateway (`AIA_GATEWAY_CLIENT_ID`, `AIA_GATEWAY_CLIENT_SECRET`, `AIA_GATEWAY_BASE_URL`), or `REALLM_BASE_URL` with `REALLM_API_KEY`. Set `SSL_VERIFY=false` only when the gateway certificate cannot be verified.
Headless Playwright does not need a desktop session. The desktop driver does need an X11 `DISPLAY`. Wayland alone is not enough.
## Run a specification
`aqe run` and `aqe serve` take no flags. Set the command parameters in `.env`.

```bash
uv run aqe run
```

`SPEC_PATH` is the specification file. `aqe run` starts the sample registration app on `SUT_PORT`, plans with the chat model in `.env`, drives browser steps with headless Chromium, desktop steps with PyAutoGUI, and checks CLI steps inside the Docker sandbox. The command prints the path to `runs/<id>/report.json`. Exit status is `0` when `verdict` is `pass`.

If Chromium, an X11 display for a desktop step, Docker, or the sandbox image is missing, the run is rejected before any step executes. A desktop step still requires an X11 display.

## Serve the agent
```bash
uv run aqe serve
```
The process binds `HOST` and `PORT` from `.env` (`127.0.0.1:8000` by default) with no authentication. Reports are written under `RUNS_DIR`.
- `http://127.0.0.1:8000/` submits a specification and lists runs. Chips show whether browser, desktop, and sandbox are available on this host.
- `http://127.0.0.1:8000/runs/<id>` shows each step as it moves from pending to running to passed, failed, retrying, error, or skipped.
- `GET /healthz` stays healthy even when optional drivers are absent.
- `GET /v1/capabilities` reports `browser`, `desktop`, and `sandbox`.
### HTTP
```bash
curl -s -X POST http://127.0.0.1:8000/v1/runs \
  -H 'content-type: application/json' \
  -d '{"specification":"# check\n\nCLI: Read the webhook\nAssertion: json:user=ada\n"}'
```
A blank or oversized specification returns `400` and does not create a run. A valid body returns `202` with an id. Poll `GET /v1/runs/<id>` until `ready` is true, then read `report`. `ready` is true for `completed`, `failed`, `rejected`, and `canceled`. While the run is `submitted` or `working`, `report` is null.
`GET /v1/runs/<id>/events` is a server-sent stream of the same snapshots. The stream ends on the first snapshot where `ready` is true.
### A2A
Discover the agent at `GET /.well-known/agent-card.json`. Send the specification as the text of a `SendMessage` call. The task id is the run id. `SendMessage` returns while the task is `submitted` or `working`. Poll `GetTask` until `status.state` is `TASK_STATE_COMPLETED`, `TASK_STATE_FAILED`, `TASK_STATE_REJECTED`, or `TASK_STATE_CANCELED`, then read the JSON artifact. Send `A2A-Version: 1.0`. A message with no text or file part is invalid and does not create a task.
## Report
Every finished run writes the same `TestReport` object to `runs/<id>/report.json`, to `report` on the HTTP snapshot, and to the A2A artifact.
`verdict` is the result to branch on:
| Verdict | Meaning |
| --- | --- |
| `pass` | Every step assertion passed. |
| `fail` | A step ran and its assertion did not hold. |
| `error` | The harness broke after start, for example the sandbox failed to launch. The result is inconclusive. |
| `rejected` | The text could not become a plan, or the host is missing a driver the plan needs. |
| `canceled` | The run was canceled before it finished. |
A pass has `verdict == "pass"`. `specification` is the testing request that was submitted. `reason_code` explains a non-pass (`assertion_failed`, `sandbox_start_failed`, `browser_launch_failed`, `desktop_input_failed`, `driver_timeout`, `engine_error`, `not_a_test_plan`, `missing_capability`, or `canceled`). Each step has `assertion_passed` set to `true`, `false`, or `null`.
## Tests
```bash
uv run pytest
uv run pytest -m integration
```
`uv run pytest` injects test drivers so it does not launch Chromium or Docker. The integration test runs Playwright, Docker, and the configured chat model, and skips when the browser or sandbox is not installed.
## Layout
- `src/aqe/graph.py` is the plan, route, execute, validate, and reflect loop.
- `src/aqe/gui/` captures, grounds, and acts through Playwright or PyAutoGUI.
- `src/aqe/cli_runtime/` synthesizes a Python snippet and runs it in the sandbox.
- `src/aqe/api.py` serves the HTTP API, the event stream, the UI, and A2A.
- `examples/specs/registration.md` is the sample plan. `examples/sut/server.py` is the registration form `aqe run` starts.