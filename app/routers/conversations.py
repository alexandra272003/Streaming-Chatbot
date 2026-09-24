from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.schemas import (
    ConversationCreate,
    ConversationRead,
    ConversationWithMessages,
    SendMessageRequest,
    SendMessageResponse,
)
from app.services import chat_service

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationRead, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: ConversationCreate, session: AsyncSession = Depends(get_session)
):
    return await chat_service.create_conversation(session, payload)


@router.get("/{conversation_id}", response_model=ConversationWithMessages)
async def get_conversation(conversation_id: int, session: AsyncSession = Depends(get_session)):
    return await chat_service.get_conversation(session, conversation_id)


@router.post("/{conversation_id}/messages", response_model=SendMessageResponse)
async def send_message(
    conversation_id: int,
    payload: SendMessageRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Non-streaming baseline (Day 21): the client waits for the FULL reply
    before getting anything back. Tomorrow's SSE endpoint sends the same
    reply, but token-by-token as it's generated instead of all at once.
    """
    return await chat_service.send_message(session, conversation_id, payload.content)
