from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Conversation, Message


async def create_conversation(session: AsyncSession, title: str) -> Conversation:
    conversation = Conversation(title=title)
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    return conversation


async def get_conversation(session: AsyncSession, conversation_id: int) -> Conversation | None:
    result = await session.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()
    if conversation is not None:
        await session.refresh(conversation, attribute_names=["messages"])
    return conversation


async def add_message(
    session: AsyncSession, conversation_id: int, role: str, content: str
) -> Message:
    message = Message(conversation_id=conversation_id, role=role, content=content)
    session.add(message)
    await session.commit()
    await session.refresh(message)
    return message


async def get_all_messages(session: AsyncSession, conversation_id: int) -> list[Message]:
    result = await session.execute(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)
    )
    return list(result.scalars().all())


async def update_summary(
    session: AsyncSession, conversation: Conversation, summary: str, summarized_through_id: int
) -> None:
    conversation.summary = summary
    conversation.summarized_through_id = summarized_through_id
    await session.commit()


async def get_recent_messages(
    session: AsyncSession, conversation_id: int, limit: int
) -> list[Message]:
    """
    Most recent `limit` messages, in chronological order -- this is what
    actually gets sent to the LLM as conversation history. See Day 24
    (truncation) for why this can't just grow unbounded forever.
    """
    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.id.desc())
        .limit(limit)
    )
    messages = list(result.scalars().all())
    messages.reverse()  # back to chronological order
    return messages
