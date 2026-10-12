"""The cost router's contract: easy tasks ride the small tier, hard ones
the large, and EVERY decision is logged as one json line — plus /chat
answers carry the decision back as a `route` object.

    python3 test_router.py
Runs against the mock backend; no network, no key.
"""
import json
import os
import tempfile

_rlog = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
os.environ["LLM_ROUTER_LOG"] = _rlog   # must land before app is imported
os.environ["LLM_REQUEST_LOG"] = tempfile.NamedTemporaryFile(
    suffix=".jsonl", delete=False).name

import app as appmod  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(appmod.app)
CHAT = {"messages": [{"role": "user", "content": "say hi in five words"}]}


def chat(body):
    r = client.post("/chat", json=body)
    assert r.status_code == 200, (r.status_code, r.text[:200])
    return r.json()


def decisions():
    lines = [ln for ln in open(_rlog).read().splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def check_logged(path, tier, reason_start):
    ds = [d for d in decisions() if d["path"] == path]
    assert ds, f"no router log line for {path}"
    d = ds[-1]
    assert d["tier"] == tier, f"{path}: expected {tier}, got {d}"
    assert d["reason"].startswith(reason_start), d
    for f in ("ts", "task", "tier", "reason", "backend",
              "prompt_tokens_est", "distinct"):
        assert f in d, f"missing {f} in {d}"
    return d


# 1. easy task -> small, and the response carries the route info
r = chat(CHAT)
assert r["route"]["tier"] == "small", r["route"]
assert r["route"]["reason"] == "short routine prompt", r["route"]
assert r["route"]["backend"] == "MockBackend", r["route"]
check_logged("/chat", "small", "short routine prompt")

# 2. long prompt -> large
long = {"messages": [{"role": "user", "content": "word " * 8000}]}
r = chat(long)
assert r["route"]["tier"] == "large", r["route"]
check_logged("/chat", "large", "long prompt")

# 3. reasoning keywords -> large
hard = {"messages": [{"role": "user",
                      "content": "debug this python code and explain why "
                                 "it fails, step by step"}]}
r = chat(hard)
assert r["route"]["tier"] == "large", r["route"]
check_logged("/chat", "large", "reasoning keyword")

# 4. explicit route hint wins over the heuristics
r = chat({**CHAT, "route": "large"})
assert r["route"]["tier"] == "large", r["route"]
check_logged("/chat", "large", "explicit route hint")

# 5. extract-with-schema is precision work -> large
r = client.post("/extract", json={
    "prompt": "order for Ada, total 12.50",
    "required": ["order_id", "total"],
    "schema": {"type": "object",
               "properties": {"order_id": {"type": "string"},
                              "total": {"type": "number"}}}})
assert r.status_code == 200, (r.status_code, r.text[:200])
check_logged("/extract", "large", "extract-with-schema")

# 6. the stream endpoint logs its decision without touching the sse payload
r = client.post("/chat/stream", json=CHAT)
assert r.status_code == 200, (r.status_code, r.text[:200])
chunks = [ln for ln in r.text.splitlines() if ln.startswith("data:")]
assert chunks, "stream produced no sse data lines"
assert all(not ln.startswith("data: [route") for ln in chunks), \
    "router info must not leak into the sse payload"
d = check_logged("/chat/stream", "small", "short routine prompt")
assert d["distinct"] is False  # no separate tier backend configured here

# 7. the env override forces a tier even against the heuristics
os.environ["LLM_ROUTE"] = "small"
r = chat(long)
assert r["route"]["tier"] == "small", r["route"]
assert r["route"]["reason"] == "LLM_ROUTE env override", r["route"]
del os.environ["LLM_ROUTE"]

# 8. one log line per request, in order
assert len(decisions()) == 7, \
    f"expected 7 router log lines, got {len(decisions())}"

print("router decisions logged per request: OK")
for d in decisions():
    print(f"  {d['path']:22s} -> {d['tier']:5s}  ({d['reason']})")
