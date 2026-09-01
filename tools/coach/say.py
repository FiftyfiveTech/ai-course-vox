#!/usr/bin/env python3
"""Append a coach (Claude) reply to a chat.jsonl.

Usage: say.py <chat.jsonl path> "message"
"""
import json, sys, time

with open(sys.argv[1], "a") as f:
    f.write(json.dumps({"role": "coach", "text": sys.argv[2],
                        "ts": time.strftime("%F %T")}) + "\n")
