"""Service configuration, read once from `JEV_*` environment variables."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JEV_", frozen=True)

    vllm_url: str = "http://localhost:45100/v1"
    vllm_api_key: str = "local-dev"
    model: str = "qwen3.6-27b"
    ### 2 asks every question in both option orders and averages: removes position bias at ~2x cost.
    permutations: int = Field(1, ge=1, le=2)
    ### vLLM batches continuously; this only bounds in-flight requests from one call.
    concurrency: int = 16
    extract_max_tokens: int = 1024
    calibration: Path = Path("/data/calibration.json")
    ### Bearer key clients must send, as with TypeSafe; unset accepts any key (the SDKs require a non-empty one).
    api_key: str | None = None
    ### Names /v1/models advertises; requests may name any, the served model answers.
    aliases: tuple[str, ...] = ("jev-latest",)
    log_level: str = "INFO"
