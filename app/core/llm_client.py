from typing import AsyncIterator

from openai import AsyncOpenAI

from app.core.config import settings

_client = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)


async def get_chat_completion(messages: list[dict]) -> str:
    """
    messages: [{"role": "user"|"assistant"|"system", "content": "..."}]

    This is the ONLY place in the app that calls the LLM provider directly.
    Every other layer works with plain message dicts, never touching the
    openai client itself -- which is what makes this swappable (a different
    provider, or a mock in tests) without changing any calling code.
    """
    response = await _client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
    )
    return response.choices[0].message.content or ""


async def stream_chat_completion(messages: list[dict]) -> AsyncIterator[str]:
    """
    Same call as get_chat_completion, but with stream=True -- instead of
    waiting for the full response, the provider sends back small chunks
    ('deltas') as they're generated. This yields just the new text of each
    chunk, one at a time, so the caller can forward each piece to a client
    immediately instead of waiting for the whole reply to finish.
    """
    stream = await _client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
