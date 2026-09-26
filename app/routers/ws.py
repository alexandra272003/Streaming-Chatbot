import json

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from openai import APIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.core.errors import AppError
from app.core.llm_client import stream_chat_completion
from app.core.moderation import validate_user_input
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
    Day 23 (WebSocket) + Day 25 (reliability) combined. FastAPI resolves
    Depends(get_session) once, when the connection is accepted -- the same
    session is then reused for every message received over this
    connection's whole lifetime, matching a WebSocket's own long-lived
    shape (unlike HTTP, where a fresh session is created and torn down per
    request).
    """
    await websocket.accept()

    try:
        while True:
            raw = await websocket.receive_text()

            # Bad input is reported back to the client, not allowed to
            # kill the whole connection -- one malformed message shouldn't
            # force a full reconnect.
            try:
                data = json.loads(raw)
                content = validate_user_input(data.get("content", ""))
            except (json.JSONDecodeError, AttributeError, AppError) as exc:
                message = getattr(exc, "message", "Malformed message")
                await websocket.send_json({"type": "error", "message": message})
                continue

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
            failed = False
            assistant_id = None
            try:
                async for token in stream_chat_completion(llm_messages):
                    collected += token
                    await websocket.send_json({"type": "token", "delta": token})
            except APIError:
                failed = True
                await websocket.send_json(
                    {"type": "error", "message": "The language model provider failed to respond"}
                )
            finally:
                # Runs on normal completion, a provider failure, AND
                # implicitly on client disconnect (send_json raising) --
                # partial replies are kept in every case, mirroring the
                # SSE path's own guarantee.
                if collected:
                    assistant_message = await repo.add_message(
                        session, conversation_id, "assistant", collected
                    )
                    assistant_id = assistant_message.id

            if not failed and assistant_id is not None:
                await websocket.send_json({"type": "done", "id": assistant_id})

    except WebSocketDisconnect:
        # Client closed the tab / lost connection -- there's no socket
        # left to send anything to. A disconnect ends the WHOLE
        # connection, not just the current message, so this is only
        # caught at the outer loop level, not per-message.
        pass
