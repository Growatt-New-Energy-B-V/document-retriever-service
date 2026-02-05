"""Application configuration."""
from __future__ import annotations

import os


class Settings:
    # LLM config — all passed through to claude-agent-sdk CLI as env vars
    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
    ANTHROPIC_AUTH_TOKEN: str = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    ANTHROPIC_BASE_URL: str = os.environ.get("ANTHROPIC_BASE_URL", "")
    ANTHROPIC_DEFAULT_SONNET_MODEL: str = os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "")
    ANTHROPIC_DEFAULT_HAIKU_MODEL: str = os.environ.get("ANTHROPIC_DEFAULT_HAIKU_MODEL", "")
    ANTHROPIC_DEFAULT_OPUS_MODEL: str = os.environ.get("ANTHROPIC_DEFAULT_OPUS_MODEL", "")
    API_TIMEOUT_MS: str = os.environ.get("API_TIMEOUT_MS", "")

    # Service config
    MCP_TOOLS_URL: str = os.environ.get("MCP_TOOLS_URL", "http://mcp-tools:8001/sse")
    DATA_DIR: str = os.environ.get("DATA_DIR", "/data")
    DB_PATH: str = os.environ.get("DB_PATH", "/data/tasks.db")
    MAX_CONCURRENT_TASKS: int = int(os.environ.get("MAX_CONCURRENT_TASKS", "1"))
    AGENT_TIMEOUT: int = int(os.environ.get("AGENT_TIMEOUT", "300"))
    AGENT_MAX_RETRIES: int = int(os.environ.get("AGENT_MAX_RETRIES", "1"))
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500MB

    @property
    def has_llm_auth(self) -> bool:
        """Check if any LLM auth is configured."""
        return bool(self.ANTHROPIC_API_KEY or self.ANTHROPIC_AUTH_TOKEN)


settings = Settings()
