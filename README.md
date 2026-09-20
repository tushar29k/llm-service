# llm-service — a local LLM microservice you can develop against today

The model backend is a **mock** (no downloads, runs instantly), so you can
build the API, prompts, streaming, structured-output retries, and benchmarks
now. `HFBackend` in `model.py` is the swap for a real model.

## Run it

```bash
cd llm-service
pip install -r requirements.txt

python3 test_smoke.py   # chat, streaming, extract_json, prompt templates
python3 bench.py        # latency, time-to-first-token, tok/s
uvicorn app:app --reload
#   POST /chat         {"messages":[{"role":"user","content":"hi"}]}
#   POST /chat/stream  {"messages":[...]}            -> SSE tokens
#   POST /extract      {"prompt":"...","required":["order_id"]}
#   GET  /info
```

## The shape

```
model.py          LocalLLM: chat / chat_stream / extract_json (retry loop)
                  MockBackend (instant) / HFBackend (real transformers code)
app.py            FastAPI: /chat, /chat/stream (SSE), /extract, /info
prompts/v1/       versioned prompt templates — never inline strings
bench.py          latency + throughput without needing HTTP
config.yaml       backend: mock | hf — one-line swap to the real model
```

## Exercises (do these — this is the production learning)

1. **Going real.** `pip install torch transformers accelerate`, set
   `backend: hf` in config.yaml, re-run `bench.py`. Compare tok/s and
   time-to-first-token with the mock. What dominates latency now?
2. **VRAM maths.** Before loading a 7B model, compute weights + KV cache
   (guide 06, Part 3 §1). Does it fit your GPU? What about INT4?
3. **Prompt A/B.** Copy `prompts/v1/summarise.txt` to `prompts/v2/`,
   change the wording, and add a `?v=` param to `/chat` that picks the
   version. Log which version each request used.
4. **Break the extractor.** Make the mock return malformed JSON (edit
   `MockBackend.generate`) and watch the retry loop recover. Then make it
   *always* malformed and watch it raise after retries.
5. **Streaming proxy.** Point `/chat/stream` at a real OpenAI-compatible
   endpoint by adding an `OpenAIBackend` (same two methods). The API layer
   shouldn't change at all — that's the point of the backend interface.
6. **Load test.** Hit `/chat` with 20 concurrent requests (python
   `concurrent.futures`) and plot latency vs concurrency. Where does it
   saturate? (With the mock: nowhere interesting. With `hf`: immediately —
   that's the lesson.)
