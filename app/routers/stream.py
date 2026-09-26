import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from openai import APIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.core.errors import NotFoundError
from app.core.llm_client import stream_chat_completion
from app.core.moderation import validate_user_input
from app.repositories import conversation_repository as repo
from app.schemas import SendMessageRequest
from app.services.chat_service import build_llm_context

router = APIRouter(prefix="/conversations", tags=["streaming"])


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/{conversation_id}/messages/stream")
async def stream_message(
    conversation_id: int,
    payload: SendMessageRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    Day 22 (SSE) + Day 25 (reliability) combined: same underlying chat
    logic as the non-streaming endpoint, forwarded token-by-token, now also
    validating input up front and handling a provider failure that happens
    mid-stream (after some tokens were already sent) gracefully instead of
    just dying.
    """
    conversation = await repo.get_conversation(session, conversation_id)
    if conversation is None:
        raise NotFoundError(f"Conversation {conversation_id} not found")

    content = validate_user_input(payload.content)
    user_message = await repo.add_message(session, conversation_id, "user", content)
    llm_messages = await build_llm_context(session, conversation_id)

    async def event_generator():
        yield _sse("user_message", {"id": user_message.id, "content": content})

        collected = ""
        failed = False
        try:
            async for token in stream_chat_completion(llm_messages):
                # Disconnect/cancel handling: stop calling the provider for
                # tokens nobody will ever receive the moment the client
                # goes away -- no point generating (and paying for) output
                # into the void.
                if await request.is_disconnected():
                    break
                collected += token
                yield _sse("token", {"delta": token})
        except APIError:
            # A failure mid-stream (after some tokens already went out)
            # can't be retried transparently -- see stream_chat_completion's
            # own docstring. Surface it as an explicit error event instead.
            failed = True
        finally:
            # No yield in here: yielding during generator close (which
            # happens on client disconnect) raises RuntimeError. Awaiting a
            # DB write is fine -- it doesn't try to hand anything back to a
            # client that may no longer be listening.
            assistant_id = None
            if collected:
                assistant_message = await repo.add_message(
                    session, conversation_id, "assistant", collected
                )
                assistant_id = assistant_message.id

        # Reached only when the client is still connected -- a disconnect
        # closes the generator before execution gets back here.
        if failed:
            yield _sse(
                "error",
                {
                    "message": "The language model provider failed to respond",
                    "partial_saved": assistant_id is not None,
                },
            )
        elif assistant_id is not None:
            yield _sse("done", {"id": assistant_id})

    return StreamingResponse(event_generator(), media_type="text/event-stream")
