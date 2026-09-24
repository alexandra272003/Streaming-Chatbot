from openai import APIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.errors import NotFoundError, ProviderError
from app.core.llm_client import get_chat_completion
from app.repositories import conversation_repository as repo
from app.schemas import (
    ConversationCreate,
    ConversationRead,
    ConversationWithMessages,
    MessageRead,
    SendMessageResponse,
)


async def create_conversation(
    session: AsyncSession, payload: ConversationCreate
) -> ConversationRead:
    conversation = await repo.create_conversation(session, payload.title)
    return ConversationRead.model_validate(conversation)


async def get_conversation(
    session: AsyncSession, conversation_id: int
) -> ConversationWithMessages:
    conversation = await repo.get_conversation(session, conversation_id)
    if conversation is None:
        raise NotFoundError(f"Conversation {conversation_id} not found")
    return ConversationWithMessages.model_validate(conversation)


async def build_llm_context(session: AsyncSession, conversation_id: int) -> list[dict]:
    """
    Day 24: context budgeting. Instead of sending EVERY message ever sent
    in a conversation to the LLM (unbounded token growth -> rising cost and
    latency, eventually exceeding the model's context window), this:

    1. Always keeps the most recent `keep_recent_messages` verbatim.
    2. Once total messages exceed `max_history_messages`, folds everything
       OLDER than the recent window into a running summary via one extra
       LLM call -- so the model still has the gist of earlier conversation,
       just compressed instead of sent word-for-word.
    3. Tracks `summarized_through_id` so a message already folded into the
       summary is never re-summarized on a later call.

    Returns the actual message list to send to the LLM: an optional
    synthetic system message carrying the summary, followed by the recent
    verbatim messages.
    """
    conversation = await repo.get_conversation(session, conversation_id)
    all_messages = await repo.get_all_messages(session, conversation_id)

    recent = all_messages[-settings.keep_recent_messages :]
    recent_ids = {m.id for m in recent}
    not_yet_summarized = [
        m
        for m in all_messages
        if m.id not in recent_ids
        and (conversation.summarized_through_id is None or m.id > conversation.summarized_through_id)
    ]

    if len(all_messages) > settings.max_history_messages and not_yet_summarized:
        summary_prompt = [
            {
                "role": "system",
                "content": (
                    "Summarize the following conversation excerpt concisely, "
                    "preserving names, facts, and decisions the user would "
                    "expect to be remembered later."
                ),
            }
        ]
        if conversation.summary:
            summary_prompt.append(
                {"role": "user", "content": f"Existing summary so far: {conversation.summary}"}
            )
        for m in not_yet_summarized:
            summary_prompt.append({"role": "user", "content": f"{m.role}: {m.content}"})

        try:
            new_summary = await get_chat_completion(summary_prompt)
        except APIError:
            new_summary = conversation.summary or ""  # summarization is best-effort, never fatal

        await repo.update_summary(
            session, conversation, new_summary, not_yet_summarized[-1].id
        )
        conversation.summary = new_summary  # keep local object in sync post-commit

    context: list[dict] = []
    if conversation.summary:
        context.append(
            {"role": "system", "content": f"Summary of earlier conversation: {conversation.summary}"}
        )
    context.extend({"role": m.role, "content": m.content} for m in recent)
    return context


async def send_message(
    session: AsyncSession, conversation_id: int, content: str
) -> SendMessageResponse:
    conversation = await repo.get_conversation(session, conversation_id)
    if conversation is None:
        raise NotFoundError(f"Conversation {conversation_id} not found")

    # Persist the user's message FIRST -- even if the LLM call below fails,
    # we don't lose what the user actually said.
    user_message = await repo.add_message(session, conversation_id, "user", content)

    llm_messages = await build_llm_context(session, conversation_id)

    try:
        reply_text = await get_chat_completion(llm_messages)
    except APIError as exc:
        raise ProviderError(
            "The language model provider failed to respond", details={"reason": str(exc)}
        )

    assistant_message = await repo.add_message(
        session, conversation_id, "assistant", reply_text
    )

    return SendMessageResponse(
        user_message=MessageRead.model_validate(user_message),
        assistant_message=MessageRead.model_validate(assistant_message),
    )
