from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from openai import APIConnectionError

from app.core import llm_client
from app.core.config import settings


def _fake_response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _conn_error():
    return APIConnectionError(request=httpx.Request("POST", "http://test"))


async def test_completion_retries_transient_error_then_succeeds(monkeypatch):
    monkeypatch.setattr(settings, "llm_retry_backoff_seconds", 0)
    create = AsyncMock(side_effect=[_conn_error(), _fake_response("ok")])
    monkeypatch.setattr(llm_client._client.chat.completions, "create", create)

    assert await llm_client.get_chat_completion([{"role": "user", "content": "hi"}]) == "ok"
    assert create.await_count == 2


async def test_completion_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(settings, "llm_retry_backoff_seconds", 0)
    monkeypatch.setattr(settings, "llm_max_retries", 2)
    create = AsyncMock(side_effect=_conn_error())
    monkeypatch.setattr(llm_client._client.chat.completions, "create", create)

    try:
        await llm_client.get_chat_completion([{"role": "user", "content": "hi"}])
        assert False, "expected APIConnectionError"
    except APIConnectionError:
        pass
    assert create.await_count == 3  # 1 attempt + 2 retries


async def test_stream_failure_midway_persists_partial_and_emits_error(client, monkeypatch):
    async def failing_stream(messages):
        yield "Hel"
        raise _conn_error()

    monkeypatch.setattr("app.routers.stream.stream_chat_completion", failing_stream)

    conv = (await client.post("/conversations", json={"title": "Chat"})).json()

    events = []
    async with client.stream(
        "POST", f"/conversations/{conv['id']}/messages/stream", json={"content": "hi"}
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                events.append(line[7:])

    assert "error" in events
    assert "done" not in events

    messages = (await client.get(f"/conversations/{conv['id']}")).json()["messages"]
    assistant = [m for m in messages if m["role"] == "assistant"]
    assert len(assistant) == 1
    assert assistant[0]["content"] == "Hel"


async def test_whitespace_only_message_rejected(client):
    conv = (await client.post("/conversations", json={"title": "Chat"})).json()
    resp = await client.post(
        f"/conversations/{conv['id']}/messages/stream", json={"content": "   "}
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "input_rejected"
