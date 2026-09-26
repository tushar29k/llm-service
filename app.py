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

llm = LocalLLM()   # load once at startup — model init is expensive,
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
    return {"backend": backend, "model": llm.cfg.get("model_id"),
            "params": params, "loaded": loaded}


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

