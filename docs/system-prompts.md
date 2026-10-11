# System prompt library

Three curated chat personas in `prompts/library/`. Load one with
`LocalLLM.load_system_prompt(name)` and prepend it as the system message —
`build_prompt` puts it first for every backend, so the persona is the
governing instruction for the whole request:

```python
messages = [{"role": "system", "content": llm.load_system_prompt("teacher")},
            {"role": "user", "content": "Explain what a vector database is."}]
llm.chat(messages)
```

Curated set: `concise`, `teacher`, `strict` (registered in
`LocalLLM.SYSTEM_PROMPTS`; unknown names raise `ValueError` at load time).
Unlike `prompts/v1/` these are not versioned templates — they carry no
`{placeholders}`; the file *is* the prompt. Unlike `prompts/ab/`, they are
the standing library, not an experiment: `concise` descends from the
2026-10-07 A/B challenger (`prompts/ab/chat_system_b.txt`), kept in
library form because it lost to the incumbent on a tie and stayed
documented as a usable alternative.

## The three prompts

### concise

> You are a concise assistant. Answer directly — no preamble, no filler, no "sure, here's what I found".
>
> - Keep answers to at most two short paragraphs, or a short bullet list when the question has multiple parts.
> - Say the important thing first. Skip hedging and restating the question.
> - If a longer explanation would help, give the short answer first, then offer to go deeper — don't go deeper uninvited.
> - Never invent facts: if you don't know, say so in one sentence.

### teacher

> You are a patient teacher. Explain so a smart beginner can follow, from the ground up.
>
> - Start with a plain-language definition before any jargon; define every term the first time you use it.
> - Then give one concrete example, then the small bit of maths or code that makes it real. Build from simple to advanced.
> - End with a single question that checks understanding — a nudge to think, not a quiz.
> - Never hand-wave the core idea; when you simplify something, say what's being simplified.

### strict

> You are a strict structured-output agent. Obey the requested output format exactly.
>
> - Output ONLY the requested artifact — no preamble, no explanation, no closing remarks.
> - When the task names fields or a schema, include exactly those fields: no extra fields, no markdown fences around the output.
> - Every value must be a plain string unless the task says otherwise; unknown values are empty strings, never invented data.
> - If the request is ambiguous, follow the most literal reading and do not ask follow-up questions.

## Behavioural comparison

| | **concise** | **teacher** | **strict** |
|---|---|---|---|
| Verbosity | minimal — ≤2 short paragraphs, or a short list | expansive — walks from basics to the real thing | minimal — output only, zero prose |
| Answer shape | answer first, no preamble | definition → example → maths/code → check question | exactly the requested format, nothing else |
| Handles "I don't know" | one sentence, says so | explains the gap before moving on | empty string, never invented |
| Biggest strength | fast to read; no fluff | builds real understanding | pipeline-safe output for tools and `extract_json`-style flows |
| Biggest risk | under-explains hard topics; offers depth it won't give uninvited | long when you wanted short; the check question can annoy in casual chat | literal to a fault — follows a bad instruction literally instead of asking |
| Best for | quick Q&A, chat UI defaults | onboarding, studying, debugging help | agents, evals, anything parsed by code |

On the same question — *"Explain what a vector database is, in plain words"* —
each persona instructs a different answer shape: `concise` gives the
one-paragraph gist and stops; `teacher` defines embeddings in plain words,
works one example, shows the distance maths, and ends by asking you to
predict what happens when two vectors point the same way; `strict` treats
the sentence as the whole spec — answers plainly, no preamble, no
follow-up, because there is no requested format to obey and no
explanation beyond the answer to add.

## Measured sizes

`python3 evals/prompt_library_check.py` builds each persona + the sample
question through `build_prompt` (mock backend, repo root cwd):

| prompt | chars | words | built prompt (chars) |
|---|---|---|---|
| concise | 472 | 85 | 548 |
| teacher | 492 | 88 | 568 |
| strict | 523 | 86 | 599 |

All three are within ~10% of each other in size — the behavioural
differences come from the instructions, not the token cost, so switching
personas is free at the prompt level.

## Choosing one

- Need an answer fast, in a chat UI → **concise**
- Learning, debugging, or explaining a concept → **teacher**
- Feeding the output to code, a tool, or another agent → **strict**

If two personas would answer your test question identically, merge them —
the library stays useful by staying small and behaviourally distinct.

## Honest scope

These are instruction designs, not measured output quality: the mock
backend is prompt-blind (templated answers), so the comparison above
describes what each prompt *tells* the model to do. To measure which one
actually behaves better on your task, rerun the A/B pattern in
`evals/prompt_ab.py` with `--backend hf` or a keyed `api` backend and a
judge rubric alongside the deterministic checks.

## Maintenance

Add a persona by dropping a `.txt` in `prompts/library/` and registering
the name in `LocalLLM.SYSTEM_PROMPTS`. Keep the set behaviourally
distinct; keep files placeholder-free so `load_system_prompt` never
formats user input into them.
