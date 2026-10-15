# ADR 0003 — cache choice

Decided 2026-10-15.

## Context

Model calls are the expensive thing this service does — real backends
cost tokens and seconds, and the demo traffic repeats itself (people
resubmit the same prompt, retry the same extract). The service needed a
cache, but there are two failure modes to avoid: serving a *wrong* answer
(stale or mismatched) is worse than serving none, and the deploy target
is Render's free tier, which has no Redis and forgets everything on a
sleep cycle. Whatever we picked had to be a plain in-process data
structure with zero new dependencies.

## Decision

Exact-first, two tiers in `_PromptCache` (app.py, shipped 2026-10-02):

- **Exact tier (always on):** key = `sha256(endpoint, backend, model,
  messages, generation params)`. A repeated combo returns the stored
  response with `"cached": true` and never touches the model. Putting
  the backend and model in the key means a config swap can't serve the
  other setup's answers — and the router passes the tier's backend, so
  small and large tier answers never share an entry.
- **Semantic tier (opt-in, off by default):** near-duplicate prompts
  match by cosine similarity over a stdlib-only hashing embedding
  (bag of hashed tokens, no libraries). Threshold and dim live in
  `config.yaml` `cache.semantic` — `enabled: false` is the default, and
  the code is marked **# SWAP**: plug a real embedder here for
  production; the hashing one just proves the plumbing works.

Expiry is TTL + LRU: entries die after `ttl_seconds` (default 1h) and
the oldest-read entry is evicted past `max_entries` (default 500).
Everything is a locked dict — thread-safe under uvicorn's threadpool,
no background sweep thread. Stats (`hits`, `misses`, `semantic_hits`,
`hit_rate`, `size`) ride on `/info.cache`, and the demo UI shows a
`cache XX%` badge off it.

Deliberately not cached: `/chat/stream` and `/v1` stream mode (a stream
has no full payload to store — see ADR 0002), and nothing on the
graceful-drain or probe paths.

## Consequences

Good: repeated demo prompts return instantly — the live Render deploy
measured `hit_rate 0.83` right after the commit. Zero dependencies, zero
infra; the mock-backed test suite proves exact hit, semantic
near-duplicate hit, and fresh-prompt miss (`test_cache.py`). Exact-first
means the always-on tier is *correct by construction* — the semantic
tier's fuzzier matches stay opt-in, gated on the flag.

Cost: in-process means the cache dies with the container — on Render's
free tier it warms back up from real traffic within minutes, so this is
a demo-acceptable loss, not a bug. No Redis, so no shared cache across
replicas; if the service ever runs two instances, hit rates halve
roughly and nobody's answer goes wrong. The hashing embedding is weak
on purpose — real semantic search needs a real embedder, which is the
documented SWAP.

Alternatives considered: Redis — rejected, another service to run for a
free-tier demo; the config still supports bolting one on (swap
`_PromptCache`'s dict for a client). Caching at the `LocalLLM` level
instead of the endpoint — rejected, the endpoints know the params that
belong in the key; caching one layer down would have re-derived them.
