"""Serve the UI and a JSON endpoint backed by QuoteMatcher.

    python server.py                 # http://127.0.0.1:8765
    python server.py --port 9000 --model models/v1

GET  /            -> ui/index.html
POST /api/match   {"text": "..."} -> {"text", "author", "score"}

Standard library only, so it adds no dependencies. Requests are handled one at a
time behind a lock: the model is small and a match takes milliseconds.
"""
import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from main import MAX_CHARS, QuoteMatcher

UI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "index.html")
MAX_BODY = 16 * 1024


class Handler(BaseHTTPRequestHandler):
    matcher: QuoteMatcher = None
    lock = threading.Lock()

    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, payload):
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(UI_PATH, "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/match":
            return self._json(404, {"error": "not found"})

        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self._json(413, {"error": "request too large"})
        try:
            text = json.loads(self.rfile.read(length) or b"{}").get("text", "")
        except (json.JSONDecodeError, AttributeError):
            return self._json(400, {"error": "expected JSON like {\"text\": \"...\"}"})
        if not isinstance(text, str) or not text.strip():
            return self._json(400, {"error": "write something first"})
        if len(text.strip()) > MAX_CHARS:
            return self._json(400, {"error": f"keep it under {MAX_CHARS:,} characters"})

        try:
            with self.lock:
                results = self.matcher.match(text, k=1)
            quote, author, score = results[0]
        except Exception as e:
            self.log_error("match failed: %r", e)
            return self._json(500, {"error": "the model hit an error, see the server log"})
        self._json(200, {"text": quote, "author": author, "score": round(score, 4)})

    def log_message(self, fmt, *args):
        if self.path.startswith("/api/"):
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/v1")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"loading {args.model}...")
    Handler.matcher = QuoteMatcher(args.model)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Philo is up at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
