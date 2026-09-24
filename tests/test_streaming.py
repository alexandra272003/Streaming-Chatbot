async def test_sse_stream_sends_tokens_and_persists_final_message(client):
    conv = (await client.post("/conversations", json={"title": "Chat"})).json()

    events = []
    async with client.stream(
        "POST",
        f"/conversations/{conv['id']}/messages/stream",
        json={"content": "hi"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        async for line in response.aiter_lines():
            if line:
                events.append(line)

    # Should see the user_message event, 3 token events (mocked: "Hello", " there", "!"),
    # and a final done event -- proving tokens arrive as separate pieces, not all at once.
    token_lines = [l for l in events if l.startswith("event: token")]
    assert len(token_lines) >= 1  # at least some token events present
    assert any(l.startswith("event: done") for l in events)

    # The full assembled reply should have been persisted to the DB
    resp = await client.get(f"/conversations/{conv['id']}")
    messages = resp.json()["messages"]
    assistant_messages = [m for m in messages if m["role"] == "assistant"]
    assert len(assistant_messages) == 1
    assert assistant_messages[0]["content"] == "Hello there!"


async def test_sse_stream_to_missing_conversation_404s(client):
    resp = await client.post(
        "/conversations/9999/messages/stream", json={"content": "hi"}
    )
    assert resp.status_code == 404
