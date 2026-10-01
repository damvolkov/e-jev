"""config.systemone: how decisions are read and served."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class SystemOneSettings(BaseSettings):
    ### 2 asks every question in both option orders and averages: removes position bias at ~2x cost.
    permutations: int = Field(1, ge=1, le=2)
    ### vLLM batches continuously; this only bounds in-flight requests from one process.
    concurrency: int = Field(16, ge=1)
    ### Total probability mass a multi-token Choice may leave unread (uniform instead): its TV-distance bound. 0 reads all.
    trie_epsilon: float = Field(1e-4, ge=0, le=0.01)
    calibration: Path = Path("/data/calibration.json")
    ### Bearer key clients must send, as with TypeSafe; unset accepts any key (the SDKs require a non-empty one).
    api_key: str | None = None
    ### Names /v1/models advertises; requests may name any, the served model answers.
    aliases: tuple[str, ...] = ("jev-latest",)
