"""config.vllm: where the served model lives and how to reach it."""

from pydantic_settings import BaseSettings


class VllmSettings(BaseSettings):
    vllm_url: str = "http://localhost:45100/v1"
    vllm_api_key: str = "local-dev"
    model: str = "qwen3.6-27b"
    extract_max_tokens: int = 1024
