# Streaming Chatbot — Week 4, Day 21 (non-streaming baseline)

A chat backend that persists conversations and messages, and calls an
LLM to generate replies. This is the **non-streaming baseline** — the
client sends a message and waits for the complete reply. Tomorrow (Day 22)
adds Server-Sent Events so the same reply streams token-by-token instead.

## Why build the non-streaming version first

Streaming adds real complexity — partial responses, disconnect handling,
backpressure. Getting the actual chat logic (persistence, history,
provider calls, error handling) working and tested *without* streaming
first means Day 22 only has to add ONE new concept (SSE), not several at
once.

## Stack

Same layered pattern as every previous project: router → service →
repository. New this week: `app/core/llm_client.py`, a thin wrapper
around the OpenAI-compatible client — the **only** place in the app that
talks to the LLM provider directly.

## The conversation/message data model

Two tables: `conversations` (just an id, title, created_at) and
`messages` (belongs to a conversation, has a `role` — `user`,
`assistant`, or `system` — and `content`). Every message, from both the
user and the assistant, is stored as its own row in chronological order.

## Why the user's message is saved *before* calling the LLM

In `app/services/chat_service.py`:

```python
user_message = await repo.add_message(session, conversation_id, "user", content)
# ... only then call the LLM
reply_text = await get_chat_completion(llm_messages)
```

If the LLM call fails — timeout, rate limit, provider outage — the
user's message is already safely persisted. Nothing they typed is lost,
even if the assistant never manages to reply. This is proven directly by
`test_user_message_persisted_even_if_llm_call_would_fail`.

## Why conversation history is sent on every call, not just the latest message

LLMs are stateless between API calls — the provider has no memory of
your previous messages unless you send them again yourself. Every call
to `get_chat_completion` includes the last `max_history_messages`
messages (default 20), not just the newest one, which is what lets the
assistant "remember" earlier things you said within the same
conversation. Proven by `test_history_sent_to_llm_includes_prior_messages`.

This also sets up **Day 24's** problem directly: conversation history
can't grow unboundedly forever — every message sent costs tokens (and
therefore money and latency), and eventually exceeds the model's context
window. That's why `max_history_messages` already exists as a hard cap,
even though truncation/summarization strategy itself isn't built until
Day 24.

## Running it for real

Requires a real API key (OpenAI, or any OpenAI-compatible provider):

```bash
cp .env.example .env
# edit .env and paste in a real LLM_API_KEY
docker compose up --build
```

```bash
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

## Running tests (no API key or Docker required)

```bash
pip install -r requirements.txt
pytest -v
```

9 tests, all passing. The LLM call is mocked (`tests/conftest.py`) — the
test suite never makes a real network call or needs a real API key, same
"fast fake" philosophy as every previous project's mongomock/SQLite/
fakeredis approach.

## What's deliberately not done yet (Day 21-24 scope)

- **No timeout/retry handling for the LLM provider** — arrives Day 25
- **No auth** — anyone can create conversations and send messages
- **No load testing** — arrives Day 26

## Day 22: SSE token streaming

`POST /conversations/{id}/messages/stream` (`app/routers/stream.py`) sends the same reply as the
baseline endpoint, but forwards each token to the client as it's generated instead of waiting for
the whole thing. Implemented with a plain `StreamingResponse` and `media_type="text/event-stream"` —
no special protocol beyond keeping one HTTP response open and formatting each chunk as
`data: {...}\n\n`.

**Disconnect handling:** the generator checks `await request.is_disconnected()` on every token and
stops calling the LLM provider the moment the client goes away — no point generating (and paying for)
tokens nobody will receive. Whatever was generated before the disconnect is still persisted, so a
cancelled stream doesn't lose a partial reply.

## Day 23: WebSocket chat mode

`ws://.../ws/conversations/{id}` (`app/routers/ws.py`) is a genuinely different kind of connection
from SSE — full-duplex, meaning the client can send new messages over the *same* open connection at
any time, rather than opening a new HTTP request per message. Overkill for pure LLM token output
alone, but this is the right shape if a client also needs to send things like "stop generating" or
typing-indicator events mid-stream.

**A real bug caught and fixed while building this:** the first version imported `SessionLocal`
directly instead of using the same `get_session` dependency every HTTP route uses. That silently
bypassed test database overrides — the WebSocket route would have always hit the real configured
database, even in tests, with no error to signal it. Fixed by using `Depends(get_session)` in the
WebSocket handler too (FastAPI supports dependency injection for WebSocket routes the same as HTTP),
which is also what makes `tests/test_websocket.py` able to test it against a real, swapped-in test
database.

## Day 24: conversation history truncation/summarization

Every message ever sent in a conversation being resent to the LLM on every call would mean unbounded
token growth — rising cost and latency, eventually exceeding the model's context window entirely.
`build_llm_context()` in `app/services/chat_service.py`:

1. Always keeps the most recent `keep_recent_messages` (default 6) verbatim.
2. Once total messages exceed `max_history_messages` (default 20), folds everything *older* than that
   recent window into a running summary via one extra LLM call.
3. Tracks `summarized_through_id` on the conversation so already-summarized messages are never
   re-summarized on a later call — only genuinely new old messages get folded in each time.

This is shared by all three endpoints (baseline, SSE, WebSocket) — the context-budgeting logic lives
in exactly one place, not duplicated per streaming mode.

## The chat console UI

`chat-console.html` is a single self-contained file — open it directly in a browser (no build step,
no server needed for the UI itself). It supports all three modes (baseline, SSE, WebSocket) via a
dropdown, so you can compare them side by side: baseline waits silently then dumps the whole reply at
once; SSE and WebSocket visibly fill in token-by-token.

Just make sure the API's CORS is enabled (`app/main.py` already allows all origins for this reason)
and that `docker compose up` is running before opening the console.

## Day 25: reliability — retries, timeouts, moderation

`app/core/llm_client.py` wraps every call to the provider in `_with_retries` — exponential backoff
(`base * 2^attempt`) for genuinely transient errors only (`APITimeoutError`, `APIConnectionError`,
`RateLimitError`, `InternalServerError`). An auth error or a bad request fails identically on every
attempt, so those are deliberately *not* retried — retrying them would only waste time before the
client gets a clear, actionable error. The SDK's own built-in retries are explicitly disabled
(`max_retries=0`) so retry behavior lives in exactly one place, not two potentially-conflicting
systems.

**Retries stop the moment a stream opens.** Once `stream_chat_completion` starts yielding tokens,
a mid-stream failure is *not* retried — some tokens may already be sent to the actual client, and
silently opening a second, independent completion request would mean the client sees duplicated or
reordered text. Instead, both `stream.py` and `ws.py` catch that failure explicitly, persist
whatever partial reply was collected, and emit a proper `error` event — proven directly by
`test_stream_failure_midway_persists_partial_and_emits_error`.

`app/core/moderation.py` adds `validate_user_input`, called before anything is persisted or sent to
the LLM: rejects empty/whitespace-only messages and oversized input, with a pluggable
`moderation_hook` slot for a real content-policy check later. Validating *before* saving (not after)
keeps the database's contents meaningfully clean — junk input never becomes a permanent row.
