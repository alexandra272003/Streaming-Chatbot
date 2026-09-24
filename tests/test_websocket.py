import asyncio
import json
import os
import tempfile

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.db import Base, get_session
from app.main import app


def test_websocket_chat_full_flow(mock_llm_stream):
    """
    Uses a self-contained sqlite db + TestClient rather than the async
    'client'/'engine' fixtures, since websocket_connect() is a synchronous
    context manager -- kept isolated here rather than forcing it into the
    async fixture style used by every other test file.
    """
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    url = f"sqlite+aiosqlite:///{path}"
    engine = create_async_engine(url, future=True)

    async def setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(setup())

    SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)

    async def override_get_session():
        async with SessionLocal() as s:
            yield s

    app.dependency_overrides[get_session] = override_get_session

    try:
        with TestClient(app) as test_client:
            conv_id = test_client.post("/conversations", json={"title": "WS Chat"}).json()["id"]

            with test_client.websocket_connect(f"/ws/conversations/{conv_id}") as ws:
                ws.send_text(json.dumps({"content": "hi"}))

                user_evt = ws.receive_json()
                assert user_evt["type"] == "user_message"
                assert user_evt["content"] == "hi"

                tokens = []
                while True:
                    evt = ws.receive_json()
                    if evt["type"] == "token":
                        tokens.append(evt["delta"])
                    elif evt["type"] == "done":
                        break

                # mocked stream yields "Hello", " there", "!" -- proves tokens
                # arrived as separate pieces, not one final blob
                assert tokens == ["Hello", " there", "!"]
                assert "".join(tokens) == "Hello there!"

            # verify persistence happened correctly via the normal HTTP endpoint
            resp = test_client.get(f"/conversations/{conv_id}")
            messages = resp.json()["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"
            assert messages[1]["content"] == "Hello there!"
    finally:
        app.dependency_overrides.clear()
        asyncio.run(engine.dispose())
        os.remove(path)


def test_websocket_missing_conversation_sends_error(mock_llm_stream):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    url = f"sqlite+aiosqlite:///{path}"
    engine = create_async_engine(url, future=True)

    async def setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(setup())

    SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)

    async def override_get_session():
        async with SessionLocal() as s:
            yield s

    app.dependency_overrides[get_session] = override_get_session

    try:
        with TestClient(app) as test_client:
            with test_client.websocket_connect("/ws/conversations/9999") as ws:
                ws.send_text(json.dumps({"content": "hi"}))
                evt = ws.receive_json()
                assert evt["type"] == "error"
    finally:
        app.dependency_overrides.clear()
        asyncio.run(engine.dispose())
        os.remove(path)
