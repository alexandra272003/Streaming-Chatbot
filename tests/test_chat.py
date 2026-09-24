from app.core.errors import ProviderError


async def test_ping(client):
    resp = await client.get("/ping")
    assert resp.status_code == 200


async def test_create_conversation(client):
    resp = await client.post("/conversations", json={"title": "Test chat"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["title"] == "Test chat"
    assert "id" in body


async def test_create_conversation_default_title(client):
    resp = await client.post("/conversations", json={})
    assert resp.status_code == 201
    assert resp.json()["title"] == "New conversation"


async def test_send_message_persists_both_sides(client, mock_llm):
    conv = (await client.post("/conversations", json={"title": "Chat"})).json()

    resp = await client.post(
        f"/conversations/{conv['id']}/messages", json={"content": "Hello there"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_message"]["role"] == "user"
    assert body["user_message"]["content"] == "Hello there"
    assert body["assistant_message"]["role"] == "assistant"
    assert body["assistant_message"]["content"] == "This is a mocked assistant reply."


async def test_get_conversation_includes_all_messages(client, mock_llm):
    conv = (await client.post("/conversations", json={"title": "Chat"})).json()
    await client.post(f"/conversations/{conv['id']}/messages", json={"content": "First"})
    await client.post(f"/conversations/{conv['id']}/messages", json={"content": "Second"})

    resp = await client.get(f"/conversations/{conv['id']}")
    assert resp.status_code == 200
    messages = resp.json()["messages"]
    # 2 user messages + 2 assistant replies = 4, in chronological order
    assert len(messages) == 4
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[0]["content"] == "First"
    assert messages[2]["content"] == "Second"


async def test_send_message_to_missing_conversation_404s(client):
    resp = await client.post("/conversations/9999/messages", json={"content": "Hi"})
    assert resp.status_code == 404


async def test_get_missing_conversation_404s(client):
    resp = await client.get("/conversations/9999")
    assert resp.status_code == 404


async def test_user_message_persisted_even_if_llm_call_would_fail(client, monkeypatch):
    """
    Proves the ordering decision in chat_service.send_message: the user's
    message is saved BEFORE calling the LLM, so a provider failure never
    loses what the user actually typed.
    """
    from unittest.mock import AsyncMock

    conv = (await client.post("/conversations", json={"title": "Chat"})).json()

    failing_llm = AsyncMock(side_effect=RuntimeError("simulated provider outage"))
    monkeypatch.setattr("app.services.chat_service.get_chat_completion", failing_llm)

    # The LLM call fails with a generic exception here (not an APIError,
    # since constructing a real APIError requires a full httpx response
    # object) -- this deliberately checks that the user message was
    # persisted BEFORE the failure point, regardless of how the call fails.
    try:
        await client.post(f"/conversations/{conv['id']}/messages", json={"content": "Save me"})
    except Exception:
        pass

    resp = await client.get(f"/conversations/{conv['id']}")
    contents = [m["content"] for m in resp.json()["messages"]]
    assert "Save me" in contents


async def test_history_sent_to_llm_includes_prior_messages(client, mock_llm):
    """Proves conversation history is actually passed to the LLM, not just the latest message."""
    conv = (await client.post("/conversations", json={"title": "Chat"})).json()
    await client.post(f"/conversations/{conv['id']}/messages", json={"content": "My name is Alex"})
    await client.post(f"/conversations/{conv['id']}/messages", json={"content": "What is my name?"})

    # second call's LLM invocation should have seen: msg1 (user), msg2
    # (assistant's reply to msg1), and msg3 (this new user message) = 3 total
    last_call_messages = mock_llm.call_args_list[-1].args[0]
    assert len(last_call_messages) == 3
    assert last_call_messages[0]["content"] == "My name is Alex"
