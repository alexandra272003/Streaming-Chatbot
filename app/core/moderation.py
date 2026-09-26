from typing import Callable

from app.core.errors import AppError

MAX_INPUT_CHARS = 8000


class InputRejected(AppError):
    status_code = 400
    code = "input_rejected"


# Pluggable moderation hook. Returns True if the text is acceptable.
# Left as None for now; a real implementation would call a moderation
# endpoint here. Tests can swap in a fake.
moderation_hook: Callable[[str], bool] | None = None


def validate_user_input(content: str) -> str:
    """
    Runs BEFORE anything is persisted or sent to the LLM -- validating
    first (rather than saving first and checking after) keeps the
    database's contents meaningfully clean: only content that already
    passed these rules ever gets stored as a real row at all. Returns the
    normalized (stripped) text.
    """
    text = content.strip()
    if not text:
        raise InputRejected("Message is empty")
    if len(text) > MAX_INPUT_CHARS:
        raise InputRejected(f"Message exceeds {MAX_INPUT_CHARS} characters")
    if moderation_hook is not None and not moderation_hook(text):
        raise InputRejected("Message was rejected by content policy")
    return text
