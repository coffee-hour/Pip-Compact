#!/usr/bin/env python3
"""Lightweight local server for the Pip-Compact web UI. Standard library only.

Serves index.html at / and exposes:
    GET  /api/status  -> {"ready": true, "params": <int>, "device": "cuda"|"cpu"}
    POST /api/chat    -> body {"message": "..."}; replies as Server-Sent Events:
                         event: token  data: {"text": "..."}
                         event: error  data: {"message": "..."}
                         event: done   data: {}

Generation uses the same byte-safe decoder as chat.py. Replies are never stored
server-side, so no replacement characters can leak into later turns.

Usage:
    python server.py --port 8000
    then open http://127.0.0.1:8000
"""
import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from chat import load, stream_reply

MAX_MSG = 400
HERE = os.path.dirname(os.path.abspath(__file__))

# Sampling defaults for the web UI. Kept explicit so they match chat.py's CLI defaults.
GEN_KW = dict(max_tokens=200, temp=0.8, top_k=40, top_p=0.9, rep_penalty=1.15)

MODEL = None
TOK = None
DEVICE = None
N_PARAMS = 0
LOCK = threading.Lock()  # one generation at a time on the single GPU/CPU


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep the terminal quiet

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/status":
            self._send_json(200, {"ready": MODEL is not None, "params": N_PARAMS,
                                  "device": DEVICE})
        else:
            self._send_json(404, {"error": "not found"})

    def _sse(self, event, data):
        msg = f"event: {event}\ndata: {json.dumps(data)}\n\n"
        self.wfile.write(msg.encode("utf-8"))
        self.wfile.flush()

    def do_POST(self):
        if self.path != "/api/chat":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            message = str(payload.get("message", "")).strip()[:MAX_MSG]
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "invalid JSON"})
            return
        if not message:
            self._send_json(400, {"error": "empty message"})
            return
        if MODEL is None:
            self._send_json(503, {"error": "model not loaded"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        prompt = f"User: {message}\nAssistant:"
        try:
            with LOCK:
                for chunk in stream_reply(MODEL, TOK, DEVICE, prompt, **GEN_KW):
                    self._sse("token", {"text": chunk})
            self._sse("done", {})
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away mid-stream
        except Exception as e:  # report generation failures to the page
            try:
                self._sse("error", {"message": str(e)})
            except (BrokenPipeError, ConnectionResetError):
                pass


def main():
    global MODEL, TOK, DEVICE, N_PARAMS
    ap = argparse.ArgumentParser(description="Serve Pip-Compact over HTTP")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ckpt", default=os.path.join(HERE, "out", "ckpt.pt"))
    ap.add_argument("--tokenizer", default=os.path.join(HERE, "out", "tokenizer.json"))
    a = ap.parse_args()

    MODEL, TOK, DEVICE = load(a.ckpt, a.tokenizer)
    N_PARAMS = sum(p.numel() for p in MODEL.parameters())
    print(f"loaded {N_PARAMS / 1e6:.1f}M params on {DEVICE}")
    print(f"serving on http://{a.host}:{a.port}")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
