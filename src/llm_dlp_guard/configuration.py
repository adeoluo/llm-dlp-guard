"""Service settings from ``DLP_*`` environment variables (or a ``.env`` file)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DLP_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8080
    upstream_url: str | None = None
    policy_path: str | None = None
    model_path: str | None = None
    backend: str = "auto"
    max_body_bytes: int = 4 * 1024 * 1024
    upstream_timeout: float = 120.0
