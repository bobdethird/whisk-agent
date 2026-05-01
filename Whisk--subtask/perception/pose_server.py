"""
Lightweight HTTP pose server — the bridge between your teammate's detection
system and the agent loop.

Your teammate POSTs their detected poses here.
Your agent loop GETs them via ServerPoseProvider.

Endpoints
---------
GET  /poses           → current pose map as JSON
POST /poses           → update all or some poses (partial updates OK)
GET  /poses/<name>    → single object pose
GET  /health          → server status + timestamp of last update
GET  /                → human-readable HTML dashboard

Run
---
    python perception/pose_server.py          # default port 5050
    python perception/pose_server.py --port 8080

Teammate integration (any language)
------------------------------------
    # Python
    import requests
    requests.post("http://localhost:5050/poses", json={
        "matcha_cup":   [0.302, 0.019, 0.401],
        "matcha_bowl":  [0.098, 0.021, 0.399],
    })

    # curl
    curl -X POST http://localhost:5050/poses \\
         -H "Content-Type: application/json" \\
         -d '{"matcha_cup": [0.30, 0.02, 0.40]}'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

# Allow running as `python perception/pose_server.py` from the project root
# as well as importing as a module.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from perception.mock_poses import MOCK_POSES

# ---------------------------------------------------------------------------
# Shared state (protected by a lock for thread safety)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_poses: dict[str, list[float]] = {k: list(v) for k, v in MOCK_POSES.items()}
_last_update: float | None = None   # epoch seconds of last POST
_update_count: int = 0


def _get_snapshot() -> dict[str, Any]:
    with _lock:
        return {
            "poses": {k: list(v) for k, v in _poses.items()},
            "last_update": _last_update,
            "update_count": _update_count,
        }


def _apply_update(data: dict) -> list[str]:
    """Update poses from a partial or full dict.  Returns list of updated keys."""
    global _last_update, _update_count
    updated = []
    with _lock:
        for name, coords in data.items():
            if not isinstance(coords, list) or len(coords) != 3:
                continue
            _poses[name] = [float(coords[0]), float(coords[1]), float(coords[2])]
            updated.append(name)
        if updated:
            _last_update = time.time()
            _update_count += 1
    return updated


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class PoseHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt: str, *args) -> None:
        # Suppress per-request noise; keep it quiet during agent runs
        pass

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------
    def do_GET(self) -> None:
        path = self.path.rstrip("/")

        if path == "/poses":
            snap = _get_snapshot()
            self._send_json(snap["poses"])

        elif path.startswith("/poses/"):
            name = path[len("/poses/"):]
            with _lock:
                if name in _poses:
                    self._send_json({name: list(_poses[name])})
                else:
                    self._send_json({"error": f"unknown object '{name}'"}, 404)

        elif path == "/health":
            snap = _get_snapshot()
            age = (time.time() - snap["last_update"]) if snap["last_update"] else None
            self._send_json({
                "status": "ok",
                "objects": list(snap["poses"].keys()),
                "update_count": snap["update_count"],
                "seconds_since_last_update": round(age, 1) if age else None,
            })

        elif path in ("", "/"):
            self._send_html(self._dashboard_html())

        else:
            self._send_json({"error": "not found"}, 404)

    # ------------------------------------------------------------------
    def do_POST(self) -> None:
        path = self.path.rstrip("/")

        if path != "/poses":
            self._send_json({"error": "POST only supported at /poses"}, 404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"invalid JSON: {exc}"}, 400)
            return

        if not isinstance(data, dict):
            self._send_json({"error": "body must be a JSON object"}, 400)
            return

        updated = _apply_update(data)
        self._send_json({"status": "ok", "updated": updated})

    # ------------------------------------------------------------------
    def _dashboard_html(self) -> str:
        snap = _get_snapshot()
        age  = (time.time() - snap["last_update"]) if snap["last_update"] else None
        age_str = f"{age:.1f}s ago" if age is not None else "never (showing mock defaults)"

        rows = "".join(
            f"<tr><td>{name}</td>"
            f"<td>{pose[0]:.4f}</td><td>{pose[1]:.4f}</td><td>{pose[2]:.4f}</td></tr>"
            for name, pose in snap["poses"].items()
        )

        return f"""<!DOCTYPE html>
<html>
<head>
  <title>Whisk Pose Server</title>
  <meta http-equiv="refresh" content="2">
  <style>
    body {{ font-family: monospace; padding: 2em; background: #111; color: #eee; }}
    h1 {{ color: #7fc97f; }}
    table {{ border-collapse: collapse; margin-top: 1em; }}
    th, td {{ padding: 8px 16px; border: 1px solid #444; text-align: left; }}
    th {{ background: #222; }}
    .dim {{ color: #888; font-size: 0.85em; }}
  </style>
</head>
<body>
  <h1>Whisk Pose Server</h1>
  <p class="dim">Auto-refreshes every 2s &nbsp;|&nbsp;
     Updates received: {snap['update_count']} &nbsp;|&nbsp;
     Last update: {age_str}</p>
  <table>
    <tr><th>Object</th><th>x (m)</th><th>y (m)</th><th>z (m)</th></tr>
    {rows}
  </table>
  <p class="dim" style="margin-top:2em">
    POST to <code>/poses</code> with JSON to update.<br>
    GET <code>/poses</code> for raw JSON.<br>
    GET <code>/health</code> for status.
  </p>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(port: int = 5050) -> None:
    """Start the pose server (blocking)."""
    server = HTTPServer(("0.0.0.0", port), PoseHandler)
    print(f"[PoseServer] Listening on http://localhost:{port}")
    print(f"[PoseServer] Dashboard: http://localhost:{port}/")
    print(f"[PoseServer] Poses:     http://localhost:{port}/poses")
    print(f"[PoseServer] POST new poses to http://localhost:{port}/poses")
    print("[PoseServer] Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[PoseServer] Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisk pose HTTP server")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()
    run(args.port)
