"""Fires 100 requests at the local app (mock backend, no auth keys) and
checks the JSONL request log parses and carries the expected fields."""
import json
import os
import tempfile

log = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
os.environ["LLM_REQUEST_LOG"] = log  # must land before app is imported

import app as appmod  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

appmod.limiter.enabled = False  # don't let the rate bucket 429 the flood
client = TestClient(appmod.app)

CASES = [
    ("post", "/chat",
     {"messages": [{"role": "user", "content": "say hi in five words"}]}),
    ("post", "/extract",
     {"prompt": "name: Ada, age: 36", "required": ["name", "age"]}),
    ("post", "/v1/chat/completions",
     {"messages": [{"role": "user", "content": "what is two plus two?"}],
      "model": "mock"}),
    ("post", "/chat/stream",
     {"messages": [{"role": "user", "content": "count to three"}]}),
    ("get", "/info", None),
]
for i in range(100):
    method, path, body = CASES[i % len(CASES)]
    fn = getattr(client, method)
    r = fn(path, json=body) if body is not None else fn(path)
    assert r.status_code == 200, (path, r.status_code, r.text[:200])

lines = [ln for ln in open(log).read().splitlines() if ln.strip()]
assert len(lines) == 100, f"expected 100 log lines, got {len(lines)}"

required = {"ts", "method", "path", "status", "latency_ms", "prompt_tokens",
            "completion_tokens", "total_tokens", "cost_usd", "backend",
            "key_hash", "stream"}
paths_seen = set()
for ln in lines:
    e = json.loads(ln)  # raises if the line isn't parseable json
    missing = required - set(e)
    assert not missing, f"missing fields {missing} in {ln[:120]}"
    assert isinstance(e["latency_ms"], (int, float)) and e["latency_ms"] >= 0
    assert e["status"] == 200
    assert e["backend"] == "mock"
    assert e["cost_usd"] == 0.0  # mock backend is free
    if e["prompt_tokens"] is not None or e["completion_tokens"] is not None:
        assert e["total_tokens"] == (e["prompt_tokens"] or 0) + \
            (e["completion_tokens"] or 0)
    if e["path"] in ("/chat", "/extract", "/v1/chat/completions",
                     "/chat/stream"):
        assert e["prompt_tokens"] and e["prompt_tokens"] > 0, e
    if e["path"] != "/info":
        assert e["completion_tokens"] and e["completion_tokens"] > 0, e
    paths_seen.add(e["path"])

assert {"/chat", "/extract", "/v1/chat/completions", "/chat/stream",
        "/info"} <= paths_seen, paths_seen
streams = [json.loads(ln) for ln in lines
           if json.loads(ln)["path"] == "/chat/stream"]
assert all(e["stream"] is True for e in streams), "stream flag missing"
print(f"ok — 100 requests logged, all parse, all fields present ({log})")
os.unlink(log)
