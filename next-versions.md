# llm-service — next versions plan

Current state (v1): OpenAI-style FastAPI microservice (chat, SSE streaming,
structured extraction with validate-and-retry, versioned prompts, bench)
running on a mock backend with an HF backend stub. Every item below is one
day's commit: implement it, keep smoke tests green, commit with a human-style
message.

## v2.1 — real models

- [ ] HF backend that actually loads a small model (e.g. Qwen2-0.5B); VRAM maths documented before loading — done when `backend: hf` answers via the real model
- [ ] Quantized loading option (8-bit), quality-vs-speed table in docs — done when bench shows FP16 vs INT8 rows
- [ ] vLLM backend option behind the same interface; tokens/sec comparison — done when bench reports vLLM vs HF rows
- [ ] Model warmup on startup + lazy-load config flag — done when first-request latency drops after warmup
- [ ] OpenAI-compatible `/v1/chat/completions` endpoint — done when an OpenAI client library can call it unchanged
- [ ] Quality spot-check set for extract_json across backends (10 cases) — done when evals/extract_cases.jsonl + runner exist
- [ ] Bench: TTFT / total / tok-s table across all backends committed to README — done when the table is in README

## v2.2 — production plumbing

- [ ] Rate limiting: token bucket per API key — done when 429s appear past the limit in a test
- [ ] API key auth: keys from env/config, rejected requests get 401 — done when unauthenticated curl fails
- [ ] Request logging: latency, tokens, cost estimate per request to JSONL — done when 100 requests produce a parseable log
- [ ] Caching: exact prompt cache + optional semantic cache; hit-rate in /info — done when repeated prompts report hits
- [ ] Retry with backoff + circuit breaker on backend failures — done when a flaky backend test recovers
- [ ] Graceful shutdown: drain in-flight requests before exit — done when SIGTERM waits for active streams
- [ ] Health/readiness probes: /health (alive) and /ready (model loaded) — done when /ready is 503 during load

## v2.3 — prompts & quality

- [ ] Prompt A/B harness: two versions, scored outputs, winner kept — done when the harness declares a winner on the spot-check set
- [ ] Prompt regression tests: golden prompt outputs run in CI — done when a prompt change that breaks goldens fails the test
- [ ] Structured output: full JSON-schema validation (jsonschema lib), not just required fields — done when a schema-violating output triggers retry
- [ ] Multi-turn handling: history truncation strategies (sliding window vs summarise) — done when a 50-turn conversation stays in budget
- [ ] System prompt library: 3 curated system prompts + behavioural comparison doc — done when docs/system-prompts.md compares them
- [ ] Cost router: route easy tasks to the small backend, hard ones to the large — done when the router's decisions are logged per request
- [ ] Load test: asyncio concurrent clients, p99 report — done when load_test.py prints the table

## v2.4 — ship it

- [ ] Dockerfile (CUDA + CPU variants) — done when both images serve /chat
- [ ] Compose: service + redis cache — done when `docker compose up` serves with caching on
- [ ] CI: smoke + bench regression on push — done when the workflow is green
- [ ] ADRs: backend interface design, streaming design, cache choice — done when docs/adr/ has 3 files
- [ ] OpenAPI docs polish + example clients (python, curl, JS) — done when docs/examples.md works copy-paste
- [ ] Release notes v2.0 in README with bench numbers — done when README leads with them
- [ ] "Going to prod" checklist: GPUs, autoscaling, monitoring, cost alerts — done when docs/production-checklist.md exists
