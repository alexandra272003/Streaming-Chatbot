# Streaming Chatbot

A FastAPI chat backend that persists conversations in PostgreSQL and generates replies through an OpenAI-compatible LLM (built and tested against Groq), with three interchangeable ways of delivering a reply: a plain HTTP response, Server-Sent Events, and WebSocket. Built as Week 4 of a 60-day backend + GenAI sprint (Day 21–25).

## What this project demonstrates

- A real LLM integration wrapped in a genuine reliability layer (timeouts, exponential-backoff retries, input validation) — not just a raw API call
- Three transport strategies (HTTP, SSE, WebSocket) sharing one identical core of business logic, rather than three diverging copies
- Deliberate correctness under failure: nothing the user typed is ever lost, even when the provider fails mid-reply
- Unbounded conversation growth solved with a rolling summary instead of an unbounded token bill or a silent memory cutoff

## Stack

| Concern | Choice |
|---|---|
| API | FastAPI (async) |
| Database | PostgreSQL + SQLAlchemy (async) + Alembic migrations |
| LLM provider | Any OpenAI-compatible API (built and verified against Groq) |
| Testing | pytest + SQLite (fast, no real infra needed) |
| Container | Docker Compose (API + Postgres) |

Same router → service → repository layering as every earlier project in the sprint. New this week: `app/core/llm_client.py` — the one and only place the app talks to the LLM provider directly.

## Data model

Two tables, linked by a foreign key:

- **`conversations`** — `id`, `title`, `created_at`, plus `summary` and `summarized_through_id` (added Day 24 — see [Context budgeting](#context-budgeting-day-24) below)
- **`messages`** — `id`, `conversation_id`, `role` (`user` / `assistant` / `system`), `content`, `created_at`

Every message, from both the user and the assistant, is its own row, in chronological order.

## The three ways to send a message

All three call the exact same underlying logic (`chat_service.build_llm_context` and the same save-order guarantee) — they only differ in how the reply gets back to the client.

| Mode | Endpoint | Behavior |
|---|---|---|
| **Baseline** (Day 21) | `POST /conversations/{id}/messages` | Client waits silently; one complete JSON response once the full reply is ready |
| **SSE** (Day 22) | `POST /conversations/{id}/messages/stream` | Same request, but the connection stays open and the reply arrives as small `event: token` frames as it's generated |
| **WebSocket** (Day 23) | `ws://.../ws/conversations/{id}` | A persistent, full-duplex connection — the client can send many messages over one connection without reopening it each time |

**Why build the non-streaming version first?** Streaming adds real complexity on its own — partial responses, disconnect handling, backpressure. Getting the core chat logic (persistence, history, provider calls, error handling) working and tested *without* streaming first meant each later mode only had to add one new concept at a time, not several at once.

**Why SSE before WebSocket, and why bother with WebSocket at all?** LLM token output is inherently one-directional (server → client), which is exactly what SSE is built for — it reuses plain HTTP with no new protocol. WebSocket is genuinely more capable (full-duplex, either side can send anytime) but that capability isn't needed for pure token streaming; it's included specifically to demonstrate the second mode and to be the right shape *if* a client later needs to send something mid-stream (a "stop generating" signal, a typing indicator).

## Why the user's message is saved before the LLM is even called

```python
# app/services/chat_service.py
user_message = await repo.add_message(session, conversation_id, "user", content)
# ... only then call the LLM
reply_text = await get_chat_completion(llm_messages)
```

If the LLM call fails — timeout, rate limit, provider outage — the user's message is already safely committed. Nothing they typed is ever lost, even if the assistant never manages to reply. Proven directly by `test_user_message_persisted_even_if_llm_call_would_fail`.

## Why full conversation history is resent on every single call

LLM providers are stateless between API calls — there is no memory of prior messages unless the client resends them. Every call to the provider includes the most recent messages (bounded by `max_history_messages`, default 20), which is what creates the appearance of the assistant "remembering" earlier turns in the same conversation. Proven by `test_history_sent_to_llm_includes_prior_messages`.

This also creates the problem Day 24 solves: history can't just grow forever — every message resent costs tokens (money and latency), and eventually exceeds the model's context window entirely.

## Context budgeting (Day 24)

`build_llm_context()` in `app/services/chat_service.py`:

1. Always keeps the most recent `keep_recent_messages` (default 6) verbatim.
2. Once total messages exceed `max_history_messages` (default 20), folds everything *older* than that recent window into a running summary via one extra LLM call.
3. Tracks `summarized_through_id` on the conversation so already-summarized messages are never re-summarized on a later call — only genuinely new old messages get folded in each time.

If the summarization call itself fails, it's treated as best-effort: the existing summary (or none) is kept, and the conversation continues working normally on the unsummarized recent window. Summarization is an optimization, not core functionality — a hiccup compressing old history should never block a reply to what the user just typed.

This logic is shared by all three endpoints — context budgeting lives in exactly one place, never duplicated per transport mode.

## Reliability: retries, timeouts, input validation (Day 25)

`app/core/llm_client.py` wraps every provider call in `_with_retries`: exponential backoff (`base * 2^attempt`) for genuinely transient errors only — `APITimeoutError`, `APIConnectionError`, `RateLimitError`, `InternalServerError`. An auth error or a malformed request fails identically on every attempt, so those are deliberately *not* retried, since retrying them would only delay the client getting a clear, actionable error. The SDK's own built-in retry support is explicitly disabled (`max_retries=0`) so retry behavior lives in exactly one place, not two potentially-conflicting systems.

**Retries stop the moment a stream opens.** Once `stream_chat_completion` starts yielding tokens, a mid-stream failure is *not* retried — some tokens may already have reached the client, and silently opening a second, independent request would mean the client sees duplicated or reordered text. Instead, both `stream.py` and `ws.py` catch that failure explicitly, persist whatever partial reply was already collected, and emit an `error` event to the client — proven by `test_stream_failure_midway_persists_partial_and_emits_error`.

`app/core/moderation.py` adds `validate_user_input`, called before anything is persisted or sent to the LLM: rejects empty/whitespace-only messages and oversized input, with a pluggable `moderation_hook` slot for a real content-policy check later. Validating *before* saving — not after — keeps the database's contents meaningfully clean; rejected input never becomes a permanent row.

## Running it for real

```bash
cp .env.example .env
# edit .env and set a real LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL
docker compose up --build
curl http://127.0.0.1:8000/ping
```

Then, via `/docs` or curl:

```bash
# create a conversation
curl -X POST http://127.0.0.1:8000/conversations -H "Content-Type: application/json" -d '{"title": "Test chat"}'

# send a message (use the id from the response above)
curl -X POST http://127.0.0.1:8000/conversations/1/messages -H "Content-Type: application/json" -d '{"content": "Hello!"}'

# view the full conversation
curl http://127.0.0.1:8000/conversations/1
```

Open `chat-console.html` directly in a browser (no build step, no server needed for the UI itself) to try all three modes side by side via a dropdown — baseline waits silently then dumps the whole reply at once; SSE and WebSocket visibly fill in token-by-token. Make sure `docker compose up` is running first; CORS is already wide open in `app/main.py` for exactly this purpose.

### Groq-specific note

This project was built and verified against Groq's OpenAI-compatible endpoint (`LLM_BASE_URL=https://api.groq.com/openai/v1`). As of testing, `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` had moved to Groq's Enterprise/Contact-Sales tier and return `model_not_found` on a standard developer key. `openai/gpt-oss-20b` is confirmed working on the free tier and is the default in `.env.example`. Check [Groq's current model list](https://console.groq.com/docs/models) if you hit a `model_not_found` error — provider-hosted model availability can change independently of this code.

## Running tests (no API key or Docker required)

```bash
pip install -r requirements.txt
pytest -v
```

The LLM is fully mocked in `tests/conftest.py` — the suite never makes a real network call or needs a real API key, the same "fast fake" philosophy as every earlier project's mongomock/SQLite/fakeredis approach. Covers: baseline chat, SSE streaming, WebSocket streaming, summarization triggering and persistence, and provider failure/retry behavior.

## API reference

| Method & path | Purpose |
|---|---|
| `GET /ping` | Health check |
| `POST /conversations` | Create a conversation |
| `GET /conversations/{id}` | Get a conversation with its full message history and summary |
| `POST /conversations/{id}/messages` | Send a message, wait for the full reply (Day 21) |
| `POST /conversations/{id}/messages/stream` | Send a message, receive the reply via SSE (Day 22) |
| `ws://.../ws/conversations/{id}` | Persistent WebSocket chat connection (Day 23) |

## What's deliberately not done yet

- **No authentication** — anyone can create conversations and send messages
- **No load testing**
- **No content moderation implementation** — the hook exists (`moderation_hook` in `app/core/moderation.py`) but is unwired; a real provider-backed check would plug in here

## A real, live-found issue worth knowing about

During manual testing, the smaller free-tier model occasionally began spontaneously re-summarizing the entire conversation on nearly every turn once summarization kicked in, rather than treating the injected summary as passive background context. This looks like a smaller model misinterpreting the injected `system`-role summary as a standing instruction rather than inert context — a known class of behavior in smaller LLMs. Not fixed (summarization mechanics are proven working regardless), but worth knowing if replies start looking unexpectedly repetitive: try strengthening the system prompt's wording, or test with a larger model for the summarization call specifically.
