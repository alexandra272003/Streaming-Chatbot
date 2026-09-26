import asyncio
from typing import AsyncIterator, Awaitable, Callable, TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from app.core.config import settings

# SDK-level retries are disabled so retry behavior lives in exactly one
# place (_with_retries), where it's easy to reason about, tune, and test --
# rather than two overlapping retry systems (the SDK's own, plus a custom
# one) potentially stacking or fighting each other.
_client = AsyncOpenAI(
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
    timeout=settings.llm_timeout_seconds,
    max_retries=0,
)

# Only transient failures are worth retrying. An auth error or a bad
# request will fail identically on every attempt, so retrying those would
# only waste time and delay a clear, actionable error reaching the client.
RETRYABLE = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)

T = TypeVar("T")


async def _with_retries(call: Callable[[], Awaitable[T]]) -> T:
    """
    Exponential backoff for transient errors: attempt 0 waits base*1,
    attempt 1 waits base*2, attempt 2 waits base*4, and so on. Retrying
    instantly and repeatedly during a real outage or rate-limit window
    would just add more load to an already-struggling provider -- spacing
    retries out geometrically gives it real time to recover first.

    Used only for the *opening* of a request (before any tokens exist),
    never for an in-progress stream -- see stream_chat_completion below.
    """
    attempt = 0
    while True:
        try:
            return await call()
        except RETRYABLE:
            if attempt >= settings.llm_max_retries:
                raise
            await asyncio.sleep(settings.llm_retry_backoff_seconds * (2**attempt))
            attempt += 1


async def get_chat_completion(messages: list[dict]) -> str:
    """
    messages: [{"role": "user"|"assistant"|"system", "content": "..."}]

    This is the ONLY place in the app that calls the LLM provider directly
    for a non-streaming reply.
    """
    response = await _with_retries(
        lambda: _client.chat.completions.create(model=settings.llm_model, messages=messages)
    )
    return response.choices[0].message.content or ""


async def stream_chat_completion(messages: list[dict]) -> AsyncIterator[str]:
    """
    Yields text deltas as they arrive.

    Retry boundary: the request is retried only until the stream is open.
    Once iteration starts, a failure propagates to the caller instead of
    being retried -- some tokens may already have been yielded to (and, in
    the SSE/WebSocket routes, already sent to) the actual client. Silently
    retrying at that point would mean opening a brand-new, independent
    completion request, and the client would see duplicated or reordered
    text: the first partial reply's tokens, followed by a second reply's
    tokens stitched onto the end. So a mid-stream failure has to propagate
    up and be handled explicitly by the caller (as an error event), never
    retried transparently here.
    """
    stream = await _with_retries(
        lambda: _client.chat.completions.create(
            model=settings.llm_model, messages=messages, stream=True
        )
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
