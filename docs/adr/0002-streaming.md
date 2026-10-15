# ADR 0002 — streaming design

Decided 2026-10-15.

## Context

Users read responses faster when tokens appear as they're generated, so
`/chat/stream` and the OpenAI-compatible `/v1/chat/completions` (with
`stream: true`) both serve SSE. But streaming fights almost everything
else in the stack: the request logger needs final token counts, the
circuit breaker needs a clean failure point, the graceful drain needs to
know when a stream is *done*, and the retry loop can't replay a reply the
client already saw half of. The stream contract had to be settled before
the retry/breaker/log/drain work piled on (all shipped 2026-10-01…05).

## Decision

Server-sent events over a plain text stream, two shapes:

- `/chat/stream` emits raw token chunks: `data: <chunk>\n\n` per chunk
  from `backend.stream()`. Simple — the demo UI just appends text.
- `/v1/chat/completions` with `stream: true` speaks OpenAI's contract:
  a first chunk with `{"role": "assistant"}`, content chunks wrapped as
  `chat.completion.chunk`, a final `finish_reason: "stop"` chunk, and the
  `data: [DONE]` terminator. This is what lets the real `openai` client
  library work against the endpoint unchanged (verified 2026-09-26).

On the server side, streams are special in four ways:

1. The breaker check happens *eagerly* — `LocalLLM.stream()` is a plain
   method (not a generator) so `_breaker._enter()` runs before any SSE
   headers are sent. A dead backend is a clean 503, not a stream that
   dies mid-flight.
2. Streams never retry once a chunk has gone out. `_stream_gen` only
   retries failures that happen *before the first chunk* — replaying a
   half-sent reply would duplicate text on the client.
3. The request logger wraps the stream's body iterator and counts
   completion tokens as chunks fly by, logging once the stream completes
   via the response background hook. The shape is untouched; only the
   counting rides along.
4. Streams are explicitly excluded from the prompt cache (2026-10-02
   decision — see ADR 0003). A cached response is a full payload; there
   are no per-token timings to serve from it.

## Consequences

Good: the SSE contract survived five subsequent features without
changing — routing is logged per request but the payload is untouched,
the drain middleware holds a slot until the last chunk is sent, and the
UI's event parser stayed naive and unbroken.

Cost: MockBackend/APIBackend/VLLMBackend "stream" by word-chunking a
finished generate — their TTFT is the full latency, not a real first
token. Only hf streams genuinely (TextIteratorStreamer in a daemon
thread). The endpoints don't advertise this distinction; clients that
care about TTFT should use hf or measure it themselves.

Alternatives considered: WebSockets — rejected, SSE composes with plain
HTTP middleware, curl, and EventSource with no client library. A
replay-on-failure stream buffer — rejected, duplicate-text bugs are worse
than a clean failure; the client just reconnects.
