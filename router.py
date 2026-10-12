"""Cost router: easy tasks ride the small tier, hard ones the large.

The point is money, not cleverness — a "hi" greeting doesn't need the
biggest model. classify() picks a tier with cheap heuristics (no model
involved, so the router itself is basically free), the endpoint serves
the request from that tier's backend, and every decision is logged as
one JSON line in logs/router.jsonl — tier, reason, which backend actually
served it, and the prompt size that drove the call.

Tier -> backend mapping comes from config.yaml `router:` (or
LLM_ROUTER_SMALL / LLM_ROUTER_LARGE env) and defaults to the service's
own backend for both tiers — classification and per-request logging work
everywhere, the traffic split just stays off until you point a tier at a
different backend (e.g. small=mock for the free tier, large=api for the
real model).
"""
import json
import os
import threading
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))

_HARD_KEYWORDS = ("explain", "why", "debug", "code", "step by step",
                  "prove", "design", "compare", "analys", "reason",
                  "algorithm", "refactor", "sql", "theorem")

_LOG_LOCK = threading.Lock()


def _log_file():
    # env beats default — same pattern as the request log's LLM_REQUEST_LOG
    p = os.environ.get("LLM_ROUTER_LOG")
    if p and p.lower() not in ("off", "0", "false", "no", "disabled"):
        return p
    return os.path.join(_HERE, "logs", "router.jsonl")


def estimate_tokens(text):
    # ~4 chars per token — close enough for a routing decision, and it
    # matches the char-based budget history.py already uses
    return max(1, len(text or "") // 4)


def classify(text, task="chat", schema=False, explicit=None, cfg=None):
    """Pick small|large for one request. Cheapest checks first — the
    common short-chat case costs a dict lookup and a str.lower()."""
    cfg = cfg or {}
    text = text or ""
    if explicit in ("small", "large"):
        # the caller knows better than the heuristics — honour it
        return {"tier": explicit,
                "reason": "explicit route hint in the request body"}
    env = (os.environ.get("LLM_ROUTE") or "").strip().lower()
    if env in ("small", "large"):
        return {"tier": env, "reason": "LLM_ROUTE env override"}
    if env == "off":
        # router bypassed: everything rides the default backend
        return {"tier": "small", "reason": "router disabled (LLM_ROUTE=off)"}
    if task == "extract" and schema:
        # precision work — a sloppy tier burns the whole retry loop
        return {"tier": "large",
                "reason": "extract-with-schema needs precision"}
    tokens = estimate_tokens(text)
    long_at = int(os.environ.get("LLM_ROUTER_LONG_TOKENS",
                                 cfg.get("long_prompt_tokens", 1500)))
    if tokens > long_at:
        return {"tier": "large",
                "reason": f"long prompt (~{tokens} tokens est > {long_at})"}
    low = text.lower()
    hit = next((k for k in _HARD_KEYWORDS if k in low), None)
    if hit:
        return {"tier": "large",
                "reason": f"reasoning keyword: {hit!r}"}
    return {"tier": "small", "reason": "short routine prompt"}


def log_decision(entry):
    """One JSON line per routed request. Never raises — a full disk
    shouldn't 500 anyone's chat."""
    entry = dict(entry)
    entry.setdefault("ts", datetime.now(timezone.utc).isoformat())
    try:
        path = _log_file()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with _LOG_LOCK:
            with open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")
    except Exception:
        pass
