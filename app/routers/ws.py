import json

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.core.llm_client import stream_chat_completion
from app.repositories import conversation_repository as repo
from app.services.chat_service import build_llm_context

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/conversations/{conversation_id}")
async def chat_websocket(
    websocket: WebSocket,
    conversation_id: int,
    session: AsyncSession = Depends(get_session),
):
    """
    Day 23: WebSocket chat mode. Unlike SSE (one plain HTTP response kept
    open, server -> client only), a WebSocket is a genuinely different,
    full-duplex connection -- the client can send new messages over the
    SAME open connection at any time, without opening a new HTTP request
    each time. Overkill for pure LLM token output, but this is exactly
    the shape needed for, e.g., a client that also wants to send
    "stop generating" or presence/typing events mid-stream.

    FastAPI supports Depends() in WebSocket routes the same as HTTP routes
    -- session is resolved once when the connection is accepted and stays
    open for the whole connection's lifetime, which is exactly right for
    a WebSocket (one long-lived exchange, not one-shot request/response).
    Using Depends() here (instead of importing SessionLocal directly) is
    also what makes this endpoint testable with a swapped-in test database,
    the same way every HTTP endpoint in this project already is.

    Connection lifecycle: accept() opens it, a loop receives messages
    until the client disconnects (WebSocketDisconnect), then that
    exception is caught so the handler exits cleanly instead of crashing.
    """
    await websocket.accept()

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            content = data.get("content", "")

            conversation = await repo.get_conversation(session, conversation_id)
            if conversation is None:
                await websocket.send_json({"type": "error", "message": "Conversation not found"})
                continue

            user_message = await repo.add_message(session, conversation_id, "user", content)
            await websocket.send_json(
                {"type": "user_message", "id": user_message.id, "content": content}
            )

            llm_messages = await build_llm_context(session, conversation_id)

            collected = ""
            async for token in stream_chat_completion(llm_messages):
                collected += token
                await websocket.send_json({"type": "token", "delta": token})

            if collected:
                assistant_message = await repo.add_message(
                    session, conversation_id, "assistant", collected
                )
                await websocket.send_json({"type": "done", "id": assistant_message.id})

    except WebSocketDisconnect:
        # Client closed the tab / lost connection -- nothing to send a
        # response to anymore. No special cleanup needed beyond this catch;
        # the get_session dependency's own "async with" handles closing
        # the DB session regardless of how the handler exits.
        pass
