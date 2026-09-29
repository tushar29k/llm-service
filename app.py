"""HTTP layer over LocalLLM. Start it with:  uvicorn app:app --reload

  POST /chat         {messages, max_tokens?} -> {response}
  POST /chat/stream {messages}              -> SSE token stream
  POST /extract     {prompt, required[]}    -> JSON, validated (retries for you)
  GET  /info                                -> which backend is actually loaded
"""
try:
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
except ImportError as e:
    raise SystemExit("pip install fastapi uvicorn  (then re-run)") from e

from model import LocalLLM

import os


def _backend_choice():
    # key set (and not explicitly disabled) -> the real model; otherwise
    # the configured backend in config.yaml (mock by default)
    if os.environ.get("LLM_API_KEY") and \
            os.environ.get("LLM_AUTO", "1") not in ("0", "false", "no"):
        return "api"
    return None


llm = LocalLLM(backend=_backend_choice())   # load once at startup — model init is expensive,
                   # never do this per request
app = FastAPI(title="llm-service")


@app.post("/chat")
def chat(body: dict):
    return {"response": llm.chat(body["messages"],
                                 max_new_tokens=body.get("max_tokens", 256))}


@app.post("/chat/stream")
def chat_stream(body: dict):
    return StreamingResponse(
        (f"data: {t}\n\n" for t in llm.chat_stream(body["messages"])),
        media_type="text/event-stream")


@app.post("/extract")
def extract(body: dict):
    return llm.extract_json(body["prompt"], body["required"])


@app.get("/info")
def info():
    backend = type(llm.backend).__name__
    params, loaded = None, True
    if backend == "HFBackend":
        # lazy backend: params only exist once the model is actually loaded
        loaded = llm.backend.loaded
        if loaded:
            params = sum(p.numel() for p in llm.backend.model.parameters())
    real = backend == "APIBackend" and getattr(llm.backend, "client", None) is not None
    client = getattr(llm.backend, "client", None)
    return {"backend": backend,
            "model": client.model if real else llm.cfg.get("model_id"),
            "params": params, "loaded": loaded,
            "real_llm": real,  # True when a live API model is behind this
            "last_error": getattr(llm.backend, "last_error", None),
            "provider": client.provider if real else None}


# -- rate limiting -------------------------------------------------------------
# token bucket per api key: a cheap guardrail so one client can't hammer
# the model. keyed off the X-API-Key header, falling back to a shared
# default bucket — real api-key auth lands in a later item, this is just
# about fair usage until then.
import time as _rt_time
import math as _rt_math
import threading as _rt_threading
from fastapi.responses import JSONResponse as _JSONResponse

_rate_cfg = llm.cfg.get("rate_limit") or {}


class _TokenBucket:
    # refills `rate` tokens/sec, holds at most `capacity` (the burst)
    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.updated = _rt_time.monotonic()
        self.lock = _rt_threading.Lock()

    def take(self):
        # True when the request may go through, else seconds to wait
        now = _rt_time.monotonic()
        with self.lock:
            # top up for the time since the last request — no background thread
            self.tokens = min(self.capacity,
                              self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            wait = (1 - self.tokens) / self.rate if self.rate else 60
            return max(1, _rt_math.ceil(wait))


class _RateLimiter:
    def __init__(self, rate, capacity, enabled=True):
        self.rate = rate
        self.capacity = capacity
        self.enabled = enabled
        self._buckets = {}          # api-key value -> _TokenBucket
        self._lock = _rt_threading.Lock()

    def key_of(self, request):
        return request.headers.get("x-api-key") or "default"

    def take(self, key):
        if not self.enabled:
            return True
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = self._buckets[key] = _TokenBucket(self.rate,
                                                           self.capacity)
        return bucket.take()


limiter = _RateLimiter(_rate_cfg.get("rate_per_sec", 10.0),
                       _rate_cfg.get("burst", 30),
                       _rate_cfg.get("enabled", True))

# the demo page and /info are observability, not model spend — never throttled
_NO_LIMIT = ("/", "/info", "/docs", "/openapi.json", "/redoc")


@app.middleware("http")
async def _rate_limit(request, call_next):
    if request.url.path not in _NO_LIMIT:
        wait = limiter.take(limiter.key_of(request))
        if wait is not True:
            return _JSONResponse(
                {"detail": "rate limit exceeded — slow down and retry"},
                status_code=429, headers={"Retry-After": str(wait)})
    return await call_next(request)


# -- openai-compatible chat completions --------------------------------------
# speaks the openai request/response contract: messages in, chat.completion
# out (or an sse stream of chat.completion.chunk), so existing openai client
# code works against this endpoint unchanged — just point base_url here.
import time as _time
import uuid as _uuid
import json as _json


def _chatcmpl_id():
    return "chatcmpl-" + _uuid.uuid4().hex[:24]


def _token_count(text):
    # mock backend has no tokenizer — whitespace words are close enough
    # for usage accounting; a real tokenizer would slot in here
    return len(text.split())


@app.post("/v1/chat/completions")
def chat_completions(body: dict):
    messages = body["messages"]
    model = body.get("model", llm.cfg.get("model_id"))
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 256)
    prompt = llm.build_prompt(messages)

    if body.get("stream"):
        def _sse():
            cid, created = _chatcmpl_id(), int(_time.time())
            # first chunk carries the role, like the real api does
            yield ("data: " + _dump_chunk(cid, created, model,
                                         {"role": "assistant"}) + "\n\n")
            for piece in llm.backend.stream(prompt, max_tokens):
                yield ("data: " + _dump_chunk(cid, created, model,
                                             {"content": piece}) + "\n\n")
            yield ("data: " + _dump_chunk(cid, created, model, {},
                                         finish="stop") + "\n\n")
            yield "data: [DONE]\n\n"
        return StreamingResponse(_sse(), media_type="text/event-stream")

    content = llm.backend.generate(prompt, max_tokens, temperature)
    return {
        "id": _chatcmpl_id(), "object": "chat.completion",
        "created": int(_time.time()), "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": _token_count(prompt),
            "completion_tokens": _token_count(content),
            "total_tokens": _token_count(prompt) + _token_count(content),
        },
    }


def _dump_chunk(cid, created, model, delta, finish=None):
    return _json.dumps({
        "id": cid, "object": "chat.completion.chunk",
        "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta,
                     "finish_reason": finish}],
    })

# -- demo ui -----------------------------------------------------------------
# open / in a browser to click through the api instead of curling it.
import os as _os
from fastapi.responses import FileResponse as _FileResponse

_UI_INDEX = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "ui", "index.html")


@app.get("/", include_in_schema=False)
def _demo_ui():
    return _FileResponse(_UI_INDEX)

