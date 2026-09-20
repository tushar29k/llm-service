# llm-service

An OpenAI-style chat/completions microservice you can develop against with zero API keys.

## The idea

Say you want to build something on top of an LLM — a chat endpoint, token streaming, structured JSON extraction with retries. Normally you'd need API keys, credits, and patience while you iterate on the *plumbing*: the API shape, prompt management, retry logic, benchmarks. That's the annoying part, and it's also the part that transfers between models.

So this repo flips it: the entire service runs against a mock backend that's instant and needs no downloads. All the real engineering — the FastAPI layer, SSE streaming, the generate-validate-retry loop for structured output, versioned prompt files, latency benchmarks — is done and working. When you're ready for a real model, it's a one-line change in `config.yaml`.

## How it works

`model.py` is the heart of it. `LocalLLM` exposes three methods and hides the backend behind them:

- `chat(messages)` — plain completion.
- `chat_stream(messages)` — yields tokens one at a time (this is what powers the SSE endpoint).
- `extract_json(prompt, required)` — asks the model for JSON, validates the required fields, and if something's wrong it feeds the error back into the prompt and tries again (up to `retries` times) before raising.

There are two backends behind that interface. `MockBackend` is the fake — templated answers, instant, zero dependencies. `HFBackend` is a real Hugging Face model via `transformers`, with proper streaming through `TextIteratorStreamer` on a background thread.

`app.py` is a thin FastAPI layer: `/chat`, `/chat/stream` (server-sent events), `/extract`, and `/info` (tells you which backend is currently loaded). The model loads once at startup, never per request.

Prompts live as versioned text files under `prompts/v1/` — loaded with `load_prompt(name, **kwargs)` and formatted with plain `{placeholders}`. No prompt strings buried in code.

`bench.py` times the backend directly (no HTTP in the loop) and reports time-to-first-token, total latency, and tokens/sec for chat and streaming.

## How to run

Prerequisites: Python 3.10+ and pip. That's it for the mock backend.

```bash
cd llm-service
pip install -r requirements.txt
```

**Smoke test** — checks chat, streaming, JSON extraction, the retry loop, and prompt templates against the mock:

```bash
python3 test_smoke.py
```

You should see:

```
llm-service smoke OK
```

If anything's broken, an assertion tells you which part.

**Benchmark** — latency, time-to-first-token, and throughput:

```bash
python3 bench.py
```

You'll get three lines: `chat:` with total time and tok/s, `stream:` with first-token time and tok/s, and `extract_json:` with the parsed object. (On the mock these numbers are meaningless for model quality — they just tell you the plumbing works and how fast your machine runs the scaffolding.)

**API server:**

```bash
uvicorn app:app --reload
```

Then:

```bash
# plain chat
curl -X POST localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "hi"}]}'

# streaming (SSE — you'll see tokens arrive one by one)
curl -N -X POST localhost:8000/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "hi"}]}'

# structured extraction with validation + retries
curl -X POST localhost:8000/extract \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Extract the order.", "required": ["order_id", "items"]}'

# what's running
curl localhost:8000/info
```

## Project layout

```
model.py          LocalLLM: chat / chat_stream / extract_json (retry loop)
                  MockBackend (instant fake) / HFBackend (real transformers model)
app.py            FastAPI: /chat, /chat/stream (SSE), /extract, /info
prompts/v1/       versioned prompt templates — files, not inline strings
bench.py          latency + throughput, straight against the backend (no HTTP)
config.yaml       backend: mock | hf — the one-line swap to a real model
test_smoke.py     sanity checks for chat, streaming, extraction, retries, prompts
```

## Honest notes

- The mock backend is dumb on purpose — templated answers, no intelligence. It's a stand-in so the API, streaming, retries, and benchmarks are all real and testable without a GPU or API bill.
- Swapping to a real model is one line in `config.yaml` (`backend: hf`, plus your `model_id`), but you'll need `torch`, `transformers`, and `accelerate` installed — and realistically a GPU for anything above ~3B parameters.
- The HF streaming path uses `TextIteratorStreamer` on a background thread because `model.generate` blocks; the mock just sleeps 5ms between words to simulate token pacing.
- Nothing here has been evaluated against a real model — `bench.py` on the mock only measures your scaffolding, not model quality.
