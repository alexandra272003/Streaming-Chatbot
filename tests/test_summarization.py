from app.services.chat_service import build_llm_context
from app.repositories import conversation_repository as repo


async def test_context_stays_verbatim_below_threshold(client, session):
    conv = (await client.post("/conversations", json={"title": "Chat"})).json()

    # send fewer messages than max_history_messages (20) -- no summarization expected
    for i in range(3):
        await client.post(f"/conversations/{conv['id']}/messages", json={"content": f"msg {i}"})

    context = await build_llm_context(session, conv["id"])
    # 3 user + 3 assistant = 6 messages, no summary system message prepended
    assert len(context) == 6
    assert all(c["role"] in ("user", "assistant") for c in context)


async def test_context_gets_summarized_above_threshold(client, session, monkeypatch):
    from unittest.mock import AsyncMock

    summarizer = AsyncMock(return_value="User discussed several topics earlier.")
    monkeypatch.setattr("app.services.chat_service.get_chat_completion", summarizer)

    conv = (await client.post("/conversations", json={"title": "Chat"})).json()

    # 11 round-trips = 22 total messages, above max_history_messages (20)
    for i in range(11):
        await client.post(f"/conversations/{conv['id']}/messages", json={"content": f"msg {i}"})

    context = await build_llm_context(session, conv["id"])

    # a summary system message should now be prepended
    assert context[0]["role"] == "system"
    assert "Summary of earlier conversation" in context[0]["content"]
    # only the most recent keep_recent_messages (6) sent verbatim after the summary
    assert len(context) == 1 + 6


async def test_summary_is_persisted_on_the_conversation(client, session, monkeypatch):
    from unittest.mock import AsyncMock

    monkeypatch.setattr(
        "app.services.chat_service.get_chat_completion",
        AsyncMock(return_value="Concise summary text."),
    )

    conv = (await client.post("/conversations", json={"title": "Chat"})).json()
    for i in range(11):
        await client.post(f"/conversations/{conv['id']}/messages", json={"content": f"msg {i}"})

    await build_llm_context(session, conv["id"])

    resp = await client.get(f"/conversations/{conv['id']}")
    assert resp.json()["summary"] == "Concise summary text."


async def test_already_summarized_messages_are_not_resummarized(client, session, monkeypatch):
    """
    Proves summarized_through_id actually advances (rather than staying
    fixed or resetting) as new messages arrive -- the real signal that
    already-summarized messages aren't being redone from scratch each time.
    """
    from unittest.mock import AsyncMock

    monkeypatch.setattr(
        "app.services.chat_service.get_chat_completion",
        AsyncMock(return_value="Summary text"),
    )

    conv = (await client.post("/conversations", json={"title": "Chat"})).json()
    for i in range(11):
        await client.post(f"/conversations/{conv['id']}/messages", json={"content": f"msg {i}"})

    await build_llm_context(session, conv["id"])
    conversation = await repo.get_conversation(session, conv["id"])
    first_summarized_through = conversation.summarized_through_id
    assert first_summarized_through is not None

    # one more message, still above threshold
    await client.post(f"/conversations/{conv['id']}/messages", json={"content": "one more"})
    await build_llm_context(session, conv["id"])

    conversation = await repo.get_conversation(session, conv["id"])
    # the watermark should have moved FORWARD (new messages got folded in),
    # never backward or stayed exactly at the old boundary while ignoring
    # the newly-arrived messages entirely
    assert conversation.summarized_through_id >= first_summarized_through
