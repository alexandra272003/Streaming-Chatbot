from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://chat_user:chat_pass@localhost:5432/chatbot"

    # OpenAI-compatible: works with OpenAI directly, or any provider that
    # implements the same API shape (many do) by overriding llm_base_url.
    llm_api_key: str = "sk-placeholder-set-in-env"
    llm_base_url: str | None = None  # None = OpenAI's default endpoint
    llm_model: str = "gpt-4o-mini"

    @field_validator("llm_base_url", mode="before")
    @classmethod
    def empty_string_means_unset(cls, v):
        """
        An env var set to LLM_BASE_URL= (nothing after the =) arrives here
        as an empty string, not as "unset" -- but the openai client treats
        base_url="" very differently from base_url=None (it tries to build
        a real request URL out of nothing, failing with a confusing
        "missing protocol" error deep in httpx). Normalizing "" to None
        here means a blank/misconfigured env var falls back safely to
        OpenAI's default endpoint instead of breaking in a hard-to-diagnose
        way at request time.
        """
        if v == "":
            return None
        return v

    # Keeps a conversation from growing unbounded -- see Day 24 (truncation).
    max_history_messages: int = 20
    # Once total messages exceed max_history_messages, everything except
    # the most recent keep_recent_messages gets folded into a running
    # summary instead of sent verbatim.
    keep_recent_messages: int = 6

    # Day 25: provider reliability
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2
    llm_retry_backoff_seconds: float = 0.5

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
