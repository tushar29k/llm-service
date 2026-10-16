# API examples — copy, paste, run

Every snippet below talks to a running service. Start one first:

```bash
uvicorn app:app --port 8000
```

Swap `http://localhost:8000` for the Render URL
(`https://tushar29k-llm-service.onrender.com`) to hit the live demo.
If the server has API keys configured (`LLM_API_KEYS`), add
`-H "Authorization: Bearer <key>"` (curl) / an `Authorization` header
(python, JS) to every call — without it you get a 401.

The interactive docs live at `/docs` (Swagger UI) and the raw schema at
`/openapi.json`.

## curl

```bash
BASE=http://localhost:8000

# liveness
curl $BASE/health

# service info: backend, cache hit_rate, router tiers
curl $BASE/info

# chat — one answer
curl -X POST $BASE/chat -H 'Content-Type: application/json' -d \
  '{"messages": [{"role": "user", "content": "Say hi in five words"}]}'

# chat — streamed tokens as they are written
curl -N -X POST $BASE/chat/stream -H 'Content-Type: application/json' -d \
  '{"messages": [{"role": "user", "content": "Count to three, slowly"}]}'

# structured extraction (retries until the required fields are present)
curl -X POST $BASE/extract -H 'Content-Type: application/json' -d \
  '{"prompt": "Order 42: 3 red widgets at $9.99 each, shipped to Berlin",
    "required": ["order_id", "items", "total", "city"]}'

# OpenAI-compatible completions (drop-in for the openai client)
curl -X POST $BASE/v1/chat/completions -H 'Content-Type: application/json' -d \
  '{"model": "mock", "messages": [{"role": "user", "content": "hi"}]}'
```

## python (stdlib only — no pip needed)

```python
import json, urllib.request

BASE = "http://localhost:8000"

def post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)

print(post("/chat", {"messages": [{"role": "user", "content": "Say hi in five words"}]})["response"])
print(post("/extract", {
    "prompt": "Order 42: 3 red widgets at $9.99 each, shipped to Berlin",
    "required": ["order_id", "items", "total", "city"],
}))

# streaming: read the SSE frames as they arrive
req = urllib.request.Request(
    BASE + "/chat/stream",
    data=json.dumps({"messages": [{"role": "user", "content": "Count to three"}]}).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req) as r:
    for line in r:
        line = line.decode().strip()
        if line.startswith("data:"):
            print(line[5:].strip(), end="", flush=True)
print()
```

## JavaScript (node 18+ — global fetch, no deps)

```js
const BASE = "http://localhost:8000";

const post = (path, payload) =>
  fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).then((r) => r.json());

const chat = await post("/chat", {
  messages: [{ role: "user", content: "Say hi in five words" }],
});
console.log(chat.response);

const extracted = await post("/extract", {
  prompt: "Order 42: 3 red widgets at $9.99 each, shipped to Berlin",
  required: ["order_id", "items", "total", "city"],
});
console.log(extracted);

// streaming: parse the SSE frames as they arrive
const res = await fetch(BASE + "/chat/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ messages: [{ role: "user", content: "Count to three" }] }),
});
const reader = res.body.getReader();
const dec = new TextDecoder();
let buf = "";
for (;;) {
  const { done, value } = await reader.read();
  if (done) break;
  buf += dec.decode(value, { stream: true });
  const frames = buf.split("\n\n");
  buf = frames.pop();
  for (const f of frames)
    if (f.startsWith("data:")) process.stdout.write(f.slice(5).trim());
}
console.log();
```
