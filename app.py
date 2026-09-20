"""HTTP service over LocalLLM. Run: uvicorn app:app --reload

  POST /chat         {messages, max_tokens?} -> {response}
  POST /chat/stream {messages}              -> SSE token stream
  POST /extract     {prompt, required[]}    -> validated JSON (retry loop)
  GET  /info                                -> backend, model, params
"""
try:
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
except ImportError as e:
    raise SystemExit("pip install fastapi uvicorn  (then re-run)") from e

from model import LocalLLM

llm = LocalLLM()               # loaded ONCE at startup — never per request
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
    params = None
    if backend == "HFBackend":
        params = sum(p.numel() for p in llm.backend.model.parameters())
    return {"backend": backend, "model": llm.cfg.get("model_id"),
            "params": params}
