"""One-field registration form that records the POST body as a webhook."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs


FORM = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Register</title></head>
<body>
  <form method="post" action="/register">
    <label>Username <input name="username"></label>
    <button type="submit">Register</button>
  </form>
</body>
</html>
"""

SUCCESS = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Registered</title></head>
<body><p>registered</p></body>
</html>
"""


def serve(webhook_path: Path, port: int) -> ThreadingHTTPServer:
    webhook_path = Path(webhook_path)
    webhook_path.parent.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path not in {"/", "/register"}:
                self.send_error(404)
                return
            body = FORM.encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/register":
                self.send_error(404)
                return
            length = int(self.headers.get("content-length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            fields = parse_qs(raw)
            user = (fields.get("username") or [""])[0]
            webhook_path.write_text(json.dumps({"user": user}), encoding="utf-8")
            body = SUCCESS.encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            return

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    import os

    port = int(os.environ.get("AQE_SUT_PORT", "8765"))
    path = Path(os.environ.get("AQE_WEBHOOK_PATH", "webhook.json"))
    serve(path, port).serve_forever()


if __name__ == "__main__":
    main()
