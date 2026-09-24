import os
import tempfile
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.core.db import Base, get_session
from app.main import app


@pytest_asyncio.fixture
async def engine():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    url = f"sqlite+aiosqlite:///{path}"
    eng = create_async_engine(url, future=True)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()
    os.remove(path)


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    """
    Replaces the real LLM call with a mock everywhere it's imported --
    no real API key or network call needed to run the test suite, same
    'fast fake' philosophy as mongomock-motor, SQLite, and fakeredis in
    every earlier project this sprint. Tests a fixed, predictable reply
    unless a specific test overrides it.
    """
    mock = AsyncMock(return_value="This is a mocked assistant reply.")
    monkeypatch.setattr("app.services.chat_service.get_chat_completion", mock)
    return mock


@pytest.fixture(autouse=True)
def mock_llm_stream(monkeypatch):
    """
    Mocks the streaming variant similarly -- yields a fixed sequence of
    fake tokens instead of calling a real provider.
    """
    async def fake_stream(messages):
        for token in ["Hello", " there", "!"]:
            yield token

    monkeypatch.setattr("app.routers.stream.stream_chat_completion", fake_stream)
    monkeypatch.setattr("app.routers.ws.stream_chat_completion", fake_stream)


@pytest_asyncio.fixture
async def session(engine):
    SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with SessionLocal() as s:
        yield s


@pytest_asyncio.fixture
async def client(engine):
    SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)

    async def override_get_session():
        async with SessionLocal() as s:
            yield s

    app.dependency_overrides[get_session] = override_get_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
