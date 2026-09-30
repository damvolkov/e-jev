"""adapters.ports: what the core needs from a model server — nothing about which one."""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol, runtime_checkable

from e_jev.core.settings import Settings
from e_jev.models.reading import Reading
from e_jev.models.systemone import Json


@runtime_checkable
class Reader(Protocol):
    async def logprobs(self, prompt: str, size: int) -> Reading:
        """Log-probabilities of the first `size` option letters as the next token, in letter order."""
        ...

    async def extract(self, prompt: str, schema: dict[str, Any]) -> Json:
        """A value generated under `schema`."""
        ...

    async def health(self) -> bool: ...


type OpenReader = Callable[[Settings], AbstractAsyncContextManager[Reader]]
