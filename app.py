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

# -- demo ui -----------------------------------------------------------------
# open / in a browser to click through the api instead of curling it.
import os as _os
from fastapi.responses import FileResponse as _FileResponse

_UI_INDEX = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "ui", "index.html")


@app.get("/", include_in_schema=False)
def _demo_ui():
    return _FileResponse(_UI_INDEX)

