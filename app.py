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

from model import LocalLLM, BreakerOpenError

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
    messages = body["messages"]
    max_tokens = body.get("max_tokens", 256)
    text = _prompt_text(messages)
    key = _cache_key("chat", messages, max_tokens=max_tokens)
    cached = _prompt_cache.get(key, text)
    if cached is not None:
        cached["cached"] = True  # caller can tell this skipped the model
        return cached
    resp = {"response": llm.chat(messages, max_new_tokens=max_tokens)}
    _prompt_cache.put(key, text, resp)
    return resp


@app.post("/chat/stream")
def chat_stream(body: dict):
    return StreamingResponse(
        (f"data: {t}\n\n" for t in llm.chat_stream(body["messages"])),
        media_type="text/event-stream")


@app.post("/extract")
def extract(body: dict):
    return llm.extract_json(body["prompt"], body["required"])


# -- prompt cache ---------------------------------------------------------------
# exact-match cache in front of the model: a repeated (endpoint, backend,
# model, messages, generation params) combo returns the stored response
# without touching the backend. stats land in /info.cache — watch hit_rate
# climb. the optional semantic tier catches near-duplicates with a
# stdlib-only hashing embedding — SWAP: plug a real embedder here for
# production; this one is just good enough to prove the plumbing works.
import hashlib as _cache_hash
import json as _cache_json
import threading as _cache_threading
import time as _cache_time

_CACHE_CFG = llm.cfg.get("cache") or {}
_SEM_CFG = _CACHE_CFG.get("semantic") or {}
_CACHE_TTL = _CACHE_CFG.get("ttl_seconds", 3600)
_CACHE_MAX = _CACHE_CFG.get("max_entries", 500)
_SEM_ON = _SEM_CFG.get("enabled", False)
_SEM_THRESH = _SEM_CFG.get("threshold", 0.97)


def _hash_embed(text, dim=64):
    # bag of hashed tokens — cosine similarity without any libraries.
    # identical text scores 1.0, near-duplicates stay high, unrelated ~0
    vec = [0.0] * dim
    for tok in str(text).lower().split():
        vec[int(_cache_hash.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
    return vec


def _cosine(a, b):
    num = sum(x * y for x, y in zip(a, b))
    den = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return num / den if den else 0.0


def _prompt_text(messages):
    return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}"
                     for m in messages if isinstance(m, dict))


def _cache_key(endpoint, messages, **params):
    # backend + model are in the key so a config swap never serves
    # another setup's answers
    payload = [endpoint, type(llm.backend).__name__,
               llm.cfg.get("model_id"), messages, params]
    return _cache_hash.sha256(
        _cache_json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


class _PromptCache:
    # locked dict of key -> {resp, exp, seen, emb}; expiry checked on
    # access, overflow evicts the oldest-read entry first
    def __init__(self):
        self.enabled = bool(_CACHE_CFG.get("enabled", True))
        self.sem_enabled = bool(_SEM_ON)
        self.sem_threshold = _SEM_THRESH
        self._store = {}
        self._lock = _cache_threading.Lock()
        self.hits = 0
        self.misses = 0
        self.semantic_hits = 0

    def reset(self):
        # tests use this to start from a clean slate
        with self._lock:
            self._store.clear()
            self.hits = self.misses = self.semantic_hits = 0

    def get(self, key, prompt_text):
        if not self.enabled:
            return None
        with self._lock:
            now = _cache_time.monotonic()
            rec = self._store.get(key)
            if rec is not None:
                if rec["exp"] <= now:
                    del self._store[key]
                else:
                    rec["seen"] = now  # refresh lru-ness on the way out
                    self.hits += 1
                    return dict(rec["resp"])
            if self.sem_enabled:
                # near-duplicate path: cosine against stored embeddings
                emb = _hash_embed(prompt_text)
                best, best_sim = None, 0.0
                for k, r in self._store.items():
                    if r["exp"] <= now:
                        continue
                    sim = _cosine(emb, r["emb"])
                    if sim > best_sim:
                        best, best_sim = k, sim
                if best is not None and best_sim >= self.sem_threshold:
                    rec = self._store[best]
                    rec["seen"] = now
                    self.hits += 1
                    self.semantic_hits += 1
                    return dict(rec["resp"])
            self.misses += 1
            return None

    def put(self, key, prompt_text, response):
        if not self.enabled:
            return
        with self._lock:
            now = _cache_time.monotonic()
            # sweep the expired first, they're free entries back
            for k in [k for k, r in self._store.items() if r["exp"] <= now]:
                del self._store[k]
            if len(self._store) >= max(1, _CACHE_MAX):
                oldest = min(self._store, key=lambda k: self._store[k]["seen"])
                del self._store[oldest]
            self._store[key] = {
                "resp": response, "exp": now + _CACHE_TTL, "seen": now,
                "emb": _hash_embed(prompt_text)}

    def stats(self):
        with self._lock:
            total = self.hits + self.misses
            return {
                "enabled": self.enabled,
                "hits": self.hits,
                "misses": self.misses,
                "semantic_hits": self.semantic_hits,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
                "size": len(self._store),
            }


_prompt_cache = _PromptCache()


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
            "provider": client.provider if real else None,
            "cache": _prompt_cache.stats(),  # hits/misses/hit_rate live here
            "breaker": llm.breaker_status()}  # open/closed/half-open — the ui badge shows it


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


# -- api key auth ----------------------------------------------------------------
# real per-key auth to sit alongside the rate limiter. keys come from the
# env (LLM_API_KEYS, comma-separated) or the `auth.keys` list in
# config.yaml. enforcement only kicks in when at least one key is
# configured — empty means open access, so local dev and the keyless demo
# keep working without surprises. with keys set, anything but a matching
# key gets a 401 and never reaches the model.
_auth_cfg = llm.cfg.get("auth") or {}


def _configured_keys():
    keys = set(_auth_cfg.get("keys") or [])
    # env lets you rotate keys without touching the config file
    keys.update(k.strip() for k in os.environ.get("LLM_API_KEYS", "").split(",")
                if k.strip())
    return keys


_KEYS = _configured_keys()


def _key_from(request):
    # X-API-Key header, same as the rate limiter — or the classic
    # Authorization: Bearer scheme for openai-style clients
    key = request.headers.get("x-api-key")
    if key:
        return key
    authz = request.headers.get("authorization") or ""
    scheme, _, token = authz.partition(" ")
    if scheme.lower() == "bearer" and token:
        return token
    return None


# registered after the rate limiter, so it runs first: a 401 shouldn't
# cost the caller any rate-limit tokens
@app.middleware("http")
async def _require_api_key(request, call_next):
    if request.url.path not in _NO_LIMIT and _KEYS:
        if _key_from(request) not in _KEYS:
            return _JSONResponse({"detail": "invalid or missing api key"},
                                 status_code=401,
                                 headers={"WWW-Authenticate": "Bearer"})
    return await call_next(request)


# breaker-open is a 503, not a 500: the backend is known-down, so this is
# "try again later", not a bug. Retry-After carries the cooldown so
# clients (and the demo ui) know when it's worth coming back.
@app.exception_handler(BreakerOpenError)
async def _breaker_open(request, exc):
    return _JSONResponse({"detail": str(exc)}, status_code=503,
                         headers={"Retry-After":
                                  str(max(1, round(exc.retry_after)))})


# -- request logging -------------------------------------------------------------
# one json line per request — latency, token counts, cost estimate — appended
# to logs/requests.jsonl (or LLM_REQUEST_LOG). registered last so it wraps the
# auth + rate-limit middlewares and the latency covers the whole request path.
# written after the response goes out, so streamed replies count their tokens
# as chunks fly by and the log lands once the stream is fully sent. cheap
# locked appends, and it never raises — logging must not break the request.
from datetime import datetime as _dt, timezone as _tz
import hashlib as _log_hash
from starlette.background import BackgroundTask as _BackgroundTask

_LOG_CFG = llm.cfg.get("logging") or {}


def _log_enabled():
    v = os.environ.get("LLM_REQUEST_LOG", "")
    if v.lower() in ("off", "0", "false", "no", "disabled"):
        return False
    return bool(_LOG_CFG.get("enabled", True))


def _log_path():
    p = os.environ.get("LLM_REQUEST_LOG")
    if p and p.lower() not in ("off", "0", "false", "no", "disabled"):
        return p
    return _LOG_CFG.get("file") or _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)), "logs", "requests.jsonl")


_LOG_PATH = _log_path()
_LOG_LOCK = _rt_threading.Lock()

# $ per 1M tokens — local backends cost nothing but time; the api backend
# defaults to rough gemini-flash rates, env beats config beats default
_API_PROMPT_PRICE = float(os.environ.get(
    "LLM_COST_PROMPT_PER_1M", _LOG_CFG.get("cost_prompt_per_1m", 0.50)))
_API_COMPLETION_PRICE = float(os.environ.get(
    "LLM_COST_COMPLETION_PER_1M", _LOG_CFG.get("cost_completion_per_1m", 1.50)))


def _backend_key():
    return type(llm.backend).__name__.replace("Backend", "").lower()


def _cost_usd(prompt_tokens, completion_tokens):
    if _backend_key() != "api":
        return 0.0
    return ((prompt_tokens or 0) * _API_PROMPT_PRICE +
            (completion_tokens or 0) * _API_COMPLETION_PRICE) / 1e6


def _prompt_words(data):
    # pull the user text out of the request body for the known endpoints —
    # word count is close enough for usage accounting without a tokenizer
    msgs = data.get("messages")
    if isinstance(msgs, list):
        text = " ".join(m.get("content", "") for m in msgs
                        if isinstance(m, dict))
    else:
        text = data.get("prompt", "") or ""
    return text


def _completion_words(path, data):
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
        return usage["completion_tokens"]  # /v1/chat/completions already counted
    if "response" in data:
        return len(str(data["response"]).split())
    return len(_json.dumps(data).split())


def _sse_words(chunk):
    # count words inside sse payloads — plain token chunks (chat/stream) and
    # openai-style chat.completion.chunk json both
    if isinstance(chunk, bytes):
        chunk = chunk.decode("utf-8", "replace")
    n = 0
    for line in chunk.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        if payload.startswith("{"):
            try:
                d = _json.loads(payload)
                n += sum(len((c.get("delta") or {}).get("content", "").split())
                         for c in d.get("choices", []))
                continue
            except Exception:
                pass
        n += len(payload.split())
    return n


async def _count_stream(body_iter, counter):
    # wrap a stream's iterator, counting completion words as chunks pass
    # through — the response shape is untouched, only the counting rides along
    async for chunk in body_iter:
        if not isinstance(chunk, dict):  # dict = passthrough asgi message
            counter["n"] += _sse_words(chunk)
        yield chunk


def _write_log(entry):
    pt, ct = entry.get("prompt_tokens"), entry.get("completion_tokens")
    entry["total_tokens"] = ((pt or 0) + (ct or 0)
                             if pt is not None or ct is not None else None)
    entry["cost_usd"] = round(_cost_usd(pt, ct), 8)
    try:
        parent = _os.path.dirname(_LOG_PATH)
        if parent:
            _os.makedirs(parent, exist_ok=True)
        with _LOG_LOCK:
            with open(_LOG_PATH, "a") as f:
                f.write(_json.dumps(entry) + "\n")
    except Exception:
        pass  # a full disk shouldn't 500 anyone's chat request


@app.middleware("http")
async def _request_logger(request, call_next):
    if not _log_enabled():
        return await call_next(request)
    start = _rt_time.monotonic()
    prompt_tokens = None
    if request.method in ("POST", "PUT") and \
            "json" in (request.headers.get("content-type") or ""):
        try:
            data = _json.loads(await request.body())
            if isinstance(data, dict):
                prompt_tokens = len(_prompt_words(data).split())
        except Exception:
            prompt_tokens = None
    key = _key_from(request)
    response = await call_next(request)
    latency_ms = (_rt_time.monotonic() - start) * 1000
    # starlette hands middleware a _StreamingResponse whose body only exists
    # as an async iterator — streams are the text/event-stream ones, the rest
    # get buffered so completion tokens can be counted from the json body
    is_stream = "text/event-stream" in (response.headers.get("content-type")
                                        or "")
    entry = {
        "ts": _dt.now(_tz.utc).isoformat(),
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "latency_ms": round(latency_ms, 1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": None,
        "backend": _backend_key(),
        # sha of the key, not the key — useful for per-key usage without
        # writing secrets to disk
        "key_hash": _log_hash.sha256(key.encode()).hexdigest()[:8]
                    if key else None,
        "stream": is_stream,
    }
    if is_stream:
        # completion tokens get counted as chunks fly by; the log lands once
        # the stream is fully sent, via the response background hook
        counter = {"n": 0}
        response.body_iterator = _count_stream(response.body_iterator,
                                               counter)
        prev_bg = response.background

        async def _finalize():
            entry["completion_tokens"] = counter["n"]
            _write_log(entry)
            if prev_bg is not None:
                await prev_bg()
        response.background = _BackgroundTask(_finalize)
    else:
        chunks = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, dict):
                chunks.append(chunk)  # passthrough asgi message, not body
            else:
                chunks.append(chunk if isinstance(chunk, bytes)
                              else str(chunk).encode())

        async def _replay():
            for ch in chunks:
                yield ch
        response.body_iterator = _replay()

        body = b"".join(c for c in chunks if isinstance(c, bytes))
        if body:
            try:
                entry["completion_tokens"] = _completion_words(
                    request.url.path, _json.loads(body))
            except Exception:
                entry["completion_tokens"] = None
        _write_log(entry)
    return response


# -- graceful shutdown ---------------------------------------------------------
# SIGTERM (Render deploys, k8s, docker stop) should drain, not drop: flip a
# draining flag, answer new work with 503, and let in-flight requests —
# streaming responses especially — finish before the process exits. the wait
# is bounded by the grace period, so one stuck client can't hold a deploy
# hostage forever.
import anyio as _sh_anyio
from contextlib import asynccontextmanager as _sh_lifespan

_SHUT_CFG = llm.cfg.get("shutdown") or {}
# env beats config — one knob for the platform to tune without a redeploy
_grace_secs = float(os.environ.get("LLM_GRACEFUL_SECS",
                                   _SHUT_CFG.get("grace_seconds", 20)))

_draining = False       # flipped the moment shutdown starts
_inflight = 0           # requests currently being served; a stream counts
                        # until its last chunk is sent, not just until the
                        # response object exists
_inflight_lock = _rt_threading.Lock()  # sync endpoints run in a threadpool


def _inflight_inc():
    global _inflight
    with _inflight_lock:
        _inflight += 1


def _inflight_dec():
    global _inflight
    with _inflight_lock:
        _inflight -= 1


@_sh_lifespan
async def _drain_on_shutdown(app):
    # uvicorn runs this on SIGTERM/SIGINT — it already stopped accepting new
    # connections by now; this waits out the requests that were mid-flight
    # when the signal landed
    yield
    global _draining
    _draining = True  # stragglers on old connections get a 503, not a hang
    deadline = _rt_time.monotonic() + max(0.0, _grace_secs)
    while _rt_time.monotonic() < deadline:
        with _inflight_lock:
            left = _inflight
        if left <= 0:
            break
        await _sh_anyio.sleep(0.05)
    # grace over with requests still stuck: the process exits and they get
    # cut — the tradeoff is bounded and documented, not silent


app.router.lifespan_context = _drain_on_shutdown

# the demo page and /info are observability, not model spend — they stay up
# while draining so probes don't flap mid-deploy
_DRAIN_OK = ("/", "/info", "/docs", "/openapi.json", "/redoc")


# registered last so it runs first: a 503 during drain shouldn't cost the
# caller rate-limit tokens or a log line
@app.middleware("http")
async def _graceful_drain(request, call_next):
    if _draining and request.url.path not in _DRAIN_OK:
        return _JSONResponse(
            {"detail": "server is shutting down — retry shortly"},
            status_code=503, headers={"Retry-After": "5"})
    _inflight_inc()
    try:
        response = await call_next(request)
    except Exception:
        _inflight_dec()
        raise
    body_iter = getattr(response, "body_iterator", None)
    if body_iter is None:
        _inflight_dec()  # plain body already rendered — nothing left to wait on
        return response

    async def _tracked():
        # hold the slot until the last chunk is actually sent — a stream
        # that only built its response object isn't done yet
        try:
            async for chunk in body_iter:
                yield chunk
        finally:
            _inflight_dec()
    response.body_iterator = _tracked()
    return response


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
            for piece in llm.stream(prompt, max_tokens):
                yield ("data: " + _dump_chunk(cid, created, model,
                                             {"content": piece}) + "\n\n")
            yield ("data: " + _dump_chunk(cid, created, model, {},
                                         finish="stop") + "\n\n")
            yield "data: [DONE]\n\n"
        return StreamingResponse(_sse(), media_type="text/event-stream")

    text = _prompt_text(messages)
    key = _cache_key("chatcmpl", messages,
                     temperature=temperature, max_tokens=max_tokens)
    cached = _prompt_cache.get(key, text)
    if cached is not None:
        cached["cached"] = True  # skipped the model, just like /chat
        return cached

    content = llm.generate(prompt, max_tokens, temperature)
    resp = {
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
    _prompt_cache.put(key, text, resp)
    return resp


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

