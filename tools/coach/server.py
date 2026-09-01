#!/usr/bin/env python3
"""web-coach bridge — static pages + chat file. Stdlib only, no deps.

Usage: server.py [--dir DIR] [--port PORT]

GET  /<page>.html   serve page from DIR
GET  /chat.jsonl    conversation (page polls this)
POST /send          append {"text": ..., "tag": ...} as a user message
"""
import argparse, json, os, time
from http.server import HTTPServer, SimpleHTTPRequestHandler

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.getcwd())
ap.add_argument("--port", type=int, default=8765)
args = ap.parse_args()
ROOT = os.path.abspath(args.dir)
CHAT = os.path.join(ROOT, "chat.jsonl")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def do_POST(self):
        if self.path != "/send":
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            msg = json.loads(self.rfile.read(n))
            text = str(msg.get("text", "")).strip()
            assert text
        except Exception:
            self.send_error(400)
            return
        entry = {"role": "user", "text": text, "tag": msg.get("tag", "chat"),
                 "ts": time.strftime("%F %T")}
        with open(CHAT, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self.send_response(204)
        self.end_headers()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    open(CHAT, "a").close()
    print(f"web-coach bridge: http://127.0.0.1:{args.port} serving {ROOT}")
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
