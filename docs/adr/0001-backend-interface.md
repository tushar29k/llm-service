# ADR 0001 — one backend interface for every model

Decided 2026-10-15.

## Context

The service needs to talk to four very different model sources: a
templated fake for dev/tests (mock), a locally loaded transformers model
(hf), a vLLM-served copy of the same weights (vllm), and a hosted API
over plain HTTPS (api — Gemini/OpenRouter, because Render's free tier has
no GPU and can't run torch comfortably). On top of that, every call has to
go through the same machinery: exponential-backoff retries, the circuit
breaker, prompt loading, and history truncation. If each endpoint
(retry logic included) had to special-case four backends, the codebase
would rot the day the second real backend landed.

## Decision

All model traffic goes through `LocalLLM` in `model.py`, and `LocalLLM`
only ever talks to a backend through one small contract:

- `generate(prompt, max_new_tokens, temperature) -> str`
- `stream(prompt, max_new_tokens) -> iterator of str chunks`
- `warmup() -> None` (pay the load cost at startup when asked)
- `loaded` (property) so `/ready` knows whether traffic is safe
- optional `tokenizer` (only hf/vllm expose it — `build_prompt` uses it
  for the real chat template, the others get the cheap stand-in)

Everything resilient lives on `LocalLLM`, not on the backends:
`generate()` wraps calls in `_with_retry` (backoff + jitter + breaker),
`stream()` checks the breaker eagerly before any SSE headers go out, and
`extract_json`'s retry loop only worries about bad JSON — transient
blips were already retried inside `generate`. New backends slot in by
implementing the five methods; nothing above them changes.

Two escape hatches are deliberate, not accidental. The api backend
falls back to the mock with a visible `[model unavailable — showing
offline mock result]` note when the HTTP call fails — the demo never
500s because a key rotated. And `VLLMBackend` reuses HF's chunking style
for streaming (vLLM's offline `LLM.generate` has no public stream
iterator), accepting that its "first token" is really the full latency.

## Consequences

Good: adding vLLM was a one-day commit (2026-09-24) — no endpoint
changes, bench just got new rows. The cost router (2026-10-12) treats
tiers as *which LocalLLM instance serves the request*, so it works with
every backend for free. Tests run entirely against the mock backend,
which is why `test_smoke.py` stays instant.

Cost: the interface is a least-common-denominator. vLLM can't expose
true token streaming through it, and backends can't pass richer signals
(completions token counts, finish reasons) up through `generate` —
which is why the OpenAI-compatible endpoint counts tokens itself with
whitespace estimates instead.

Alternatives considered: a LangChain-style abstraction — rejected, it
pulls a dependency tree heavier than the service for no gain. Backend
selection is still config + env (`config.yaml` backend, `LLM_API_KEY`
auto-select) because that's one line to change and zero code.
