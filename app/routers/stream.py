import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.core.errors import NotFoundError
from app.core.llm_client import stream_chat_completion
from app.repositories import conversation_repository as repo
from app.schemas import SendMessageRequest
from app.services.chat_service import build_llm_context

router = APIRouter(prefix="/conversations", tags=["streaming"])


@router.post("/{conversation_id}/messages/stream")
async def stream_message(
    conversation_id: int,
    payload: SendMessageRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    Day 22: Server-Sent Events. Same underlying chat logic as the
    non-streaming endpoint, but the reply is forwarded to the client
    token-by-token as it's generated, instead of all at once at the end.

    SSE is one-way (server -> client) over a plain HTTP response kept open
    -- no special protocol beyond setting the right content type and
    formatting each chunk as "data: ...\\n\\n". This is why SSE, not
    WebSockets, is usually the simpler choice for LLM streaming: token
    output is inherently server-to-client only.
    """
    conversation = await repo.get_conversation(session, conversation_id)
    if conversation is None:
        raise NotFoundError(f"Conversation {conversation_id} not found")

    user_message = await repo.add_message(session, conversation_id, "user", payload.content)
    llm_messages = await build_llm_context(session, conversation_id)

    async def event_generator():
        collected = ""
        yield f"event: user_message\ndata: {json.dumps({'id': user_message.id, 'content': user_message.content})}\n\n"

        try:
            async for token in stream_chat_completion(llm_messages):
                # Disconnect/cancel handling: if the client has gone away
                # (closed the tab, cancelled the request), stop calling the
                # LLM provider for tokens nobody will ever receive -- no
                # point paying for or generating output into the void.
                if await request.is_disconnected():
                    break

                collected += token
                yield f"event: token\ndata: {json.dumps({'delta': token})}\n\n"
        finally:
            # Persist whatever was generated, even a partial reply from an
            # early disconnect -- better to keep a partial answer than lose
            # it entirely.
            if collected:
                assistant_message = await repo.add_message(
                    session, conversation_id, "assistant", collected
                )
                yield f"event: done\ndata: {json.dumps({'id': assistant_message.id})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
