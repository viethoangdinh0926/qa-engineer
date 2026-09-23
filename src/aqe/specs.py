"""Load Markdown specifications and extract a thin OpenAPI summary."""

import json
from pathlib import Path


def openapi_to_text(raw: str) -> str:
    """Turn an OpenAPI document into planner text. JSON is parsed; YAML is scanned."""
    document: dict[str, object] | None = None
    stripped = raw.lstrip()
    if stripped.startswith("{"):
        document = json.loads(raw)
    else:
        document = _scan_yaml_paths(raw)
    paths = document.get("paths") if isinstance(document, dict) else None
    if not isinstance(paths, dict) or not paths:
        return raw
    lines = ["# Operations extracted from OpenAPI", ""]
    for path, operations in paths.items():
        if not isinstance(operations, dict):
            continue
        for method, operation in operations.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                continue
            summary = method.upper()
            if isinstance(operation, dict):
                summary = str(
                    operation.get("summary")
                    or operation.get("operationId")
                    or method.upper()
                )
            lines.append(f"CLI: {method.upper()} {path} — {summary}")
            lines.append("Assertion: status:200")
    return "\n".join(lines)


def _scan_yaml_paths(raw: str) -> dict[str, object]:
    paths: dict[str, dict[str, dict[str, str]]] = {}
    current_path: str | None = None
    in_paths = False
    for line in raw.splitlines():
        if line.startswith("paths:"):
            in_paths = True
            continue
        if not in_paths:
            continue
        if line and not line.startswith(" "):
            break
        stripped = line.strip()
        if stripped.startswith("/") and stripped.endswith(":"):
            current_path = stripped[:-1]
            paths[current_path] = {}
            continue
        if current_path and stripped.endswith(":") and not stripped.startswith("/"):
            method = stripped[:-1]
            if method in {"get", "post", "put", "patch", "delete"}:
                paths[current_path][method] = {"summary": f"{method.upper()} {current_path}"}
    return {"paths": paths}


def load_spec(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        return openapi_to_text(text)
    return text
